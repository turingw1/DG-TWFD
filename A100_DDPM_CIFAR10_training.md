##  训练样本抽样策略

## 单跳跨度分布
不要只采短跳，也不要全是长跳。

推荐比例：

- **短跳**：\(s=t-\Delta\)，\(\Delta\in[1,4]\) steps，占 40%
- **中跳**：\(\Delta\in[5,16]\) steps，占 40%
- **长跳**：\(\Delta\in[17,64]\) steps，占 20%

这样兼顾局部拟合和 few-step 泛化。

## semigroup 三元组
推荐：
- 先采 \(t\)
- 再在其后随机采 \(s\)
- 再在 \(s\) 后随机采 \(u\)

并保证：
- 30% 偏短链
- 50% 中等链
- 20% 长链

## 局部 triplet
- 相邻或近邻三点即可
- 例如 \((i, i+2, i+4)\) 或 \((i, i+1, i+2)\)

---

## 训练的算法
根据上述抽样策略，从缓存轨迹中抽样训练数据，计算匹配损失、semigroup defect、时间扭曲损失，以及可选的边界损失，并按计划更新 student flow map、时间扭曲函数和边界校正器。明确的训练步骤如下，你需要根据这个算法更新训练函数和相关模块：
```python
# Teacher trajectory cache: trajectories = [{(t_i, x_ti)}]
initialize student flow map M_theta
initialize time-warp g_phi as identity
initialize optional boundary corrector B_psi
initialize defect-adaptive scheduler p_eta(t)

for each training step:

    # 1. sample a teacher trajectory
    traj = sample(trajectories)

    # 2. sample times in warped time space
    u_t, u_s, u_u = sample_times_from(p_eta)
    t = g_phi_inv(u_t)
    s = g_phi_inv(u_s)
    u = g_phi_inv(u_u)

    # 3. fetch/interpolate teacher states
    x_t = get_state(traj, t)
    x_s_T = get_state(traj, s)
    x_u_T = get_state(traj, u)

    # 4. match loss
    x_s_pred = M_theta(t, s, x_t)
    L_match = ||x_s_pred - x_s_T||^2

    # 5. semigroup defect
    x_u_pred_direct = M_theta(t, u, x_t)
    x_u_pred_comp = M_theta(s, u, x_s_pred.detach_or_not)
    L_def = ||x_u_pred_direct - x_u_pred_comp||^2

    # 6. time-warp loss using local displacement smoothness
    t3, t2, t1 = sample_local_triplet_via_u_space(g_phi)
    x_t3, x_t2, x_t1 = get_state(traj, t3), get_state(traj, t2), get_state(traj, t1)
    delta_32 = teacher_map(traj, t3, t2, x_t3)
    delta_21 = teacher_map(traj, t2, t1, x_t2)
    L_warp = ||delta_32 - delta_21||^2

    # 7. optional boundary correction
    if use_boundary and near_noise_end(t):
        x_bdry_pred = B_psi(x_tmax)
        L_bdry = ||x_bdry_pred - x_tmax_minus_delta_T||^2
    else:
        L_bdry = 0

    # 8. total loss
    L = L_match + alpha * L_def + beta * L_warp + lambda_ * L_bdry

    # 9. update student every step
    update(theta, grad(L))

    # 10. update warp every K_phi steps
    if step % K_phi == 0:
        update(phi, grad(L_def + beta * L_warp))

    # 11. update boundary only in early stage
    if use_boundary and step < warmup_end:
        update(psi, grad(L_bdry))

    # 12. refresh scheduler statistics
    update_defect_statistics()
    update_eta()
```