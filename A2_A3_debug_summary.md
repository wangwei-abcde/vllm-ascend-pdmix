# A2-A3 边云模式 MoE 通信阻塞问题调试总结

## 一、问题现象

### 环境参数
- `-dp 2 --edge-npu-count 2 --cloud-npu-count 4`
- 总 12 卡：边 2 卡 + 云 8 卡（每个 DP 云侧 4 卡，EP group = 4）
- 边 A2 → 云 A3（A2-A2 没问题，A2-A3 有问题）
- **非必现**：前 4 次 run 成功，第 5 次卡死

### 阻塞位置
第五次 run（warmup is_profile 阶段）卡点多次变化，最新卡点：

**最新卡点**：MC2 `token_dispatch` 中 `torch.npu.synchronize()`（`token_dispatcher.py` L252）
- 最后打印日志：`[HANG] moe dispatch EXIT`
- 即 `npu_moe_distribute_dispatch_v2` kernel 返回了，但随后的 `torch.npu.synchronize()` 卡住
- 说明 MC2 fused kernel 内部提交的 HCCL all2all 无法完成

**历史卡点**（已被新卡点替代）：
- 最早：ALLTOALL `_preprocess` 中 `torch.repeat_interleave`（原始 L713）
- 加 barrier 后：ALLTOALL `_preprocess` 中 `torch.npu.synchronize()`
- 删 barrier 后：MC2 dispatch 中 `torch.npu.synchronize()` ← **当前**

---

## 二、关键架构理解

### EP group 布局
- EP group 是**跨 DP** 的：`[ep_edge_ranks, ep_cloud_ranks]`
- 云侧 EP group 包含所有 DP 的云 rank：`[2,3,4,5,8,9,10,11]`（共 8 个）
- 但实际 EP 通信只在**同一 DP 内**的云 rank 间进行（DP0: 4 个，DP1: 4 个）
- 每个 DP 内部 EP 通信用 4 卡的 process group

### MC2 和 ALLTOALL 共享通信器
`parallel_state.py` 中 MC2 group 和 EP group 用相同的 rank 列表初始化：
```python
# vllm-ascend/distributed/parallel_state.py L416-421
# MC2 和 EP 都用 [ep_edge_ranks, ep_cloud_ranks]
```
ProcessGroupHCCL 对相同 rank 集合复用同一个 HCCL 通信器。

### MoE 通信类型切换
`select_moe_comm_method`（`ascend_forward_context.py`）根据 `num_tokens` vs `mc2_tokens_capacity` 选择：
- **MC2** → `num_tokens <= capacity`
- **ALLTOALL** → `num_tokens > capacity`
- A3 必须走 MC2 或 ALLTOALL，不走 ALLGATHER（ALLGATHER 仅在 A2 fallback）

第五次 run `num_tokens=1024` 超过 capacity，从 MC2 切换到 ALLTOALL。

---

## 三、已尝试的修复及结果

### 1. MC2 dispatch/combine 后加 `torch.npu.synchronize()` ✅ 保留
- **位置**：`token_dispatcher.py` MC2 的 `token_dispatch` (L252) 和 `token_combine` (L367)
- **目的**：排空 MC2 fused kernel 的计算流
- **结果**：有效但有局限（见下方分析）

### 2. ALLTOALL all_gather 后加 `torch.npu.synchronize()` ✅ 保留
- **位置**：`token_dispatcher.py` ALLTOALL `_preprocess` L676（原始）
- **目的**：all_gather_into_tensor 返回后确保数据可见

### 3. MC2 dispatch/combine 后加 `dist.barrier(group=get_ep_group().device_group)` ❌ 已移除
- **位置**：MC2 dispatch 和 combine 的 sync 之后
- **目的**：在共享 HCCL 通信器上排队 allreduce 以清空 MC2 残留操作
- **结果**：**制造了更严重的死锁** — barrier 的 allreduce 需要跨 DP 所有云 rank 参与，造成集体不匹配，HCCL 流卡死，导致后续 all_gather + `torch.npu.synchronize()` 全部阻塞
- **教训**：barrier 在跨 DP EP group 上不可用

### 4. 边侧 `.item()` NaN 检测阻塞 ⏸️ 未修改
- **位置**：`model_runner_v1.py` L5797
- **原因**：`.item()` 触发 NPU→CPU 同步，被 MoE 层未完成的 HCCL 操作阻塞

---

## 四、当前代码中的诊断日志

### token_dispatcher.py — ALLTOALL `_preprocess`
```
L663  [HANG] EP_ALLGATHER ENTER
L669  [HANG] EP_ALLGATHER canary ENTER
L673  [HANG] EP_ALLGATHER canary EXIT        ← canary all_reduce 通过
L675  gather_from_sequence_parallel_region()  → 进入 _gather_along_first_dim
L682  [HANG] EP_ALLGATHER canary2 EXIT        ← all_gather 返回值后 all_reduce
L684  torch.npu.synchronize()
L685  [HANG] EP_ALLGATHER EXIT
```
> **注意**：当前卡点已前移到 MC2 dispatch，上述 ALLTOALL 日志**没有打印**，即还没进入 ALLTOALL 就卡在 MC2 了。

### comm_utils.py — `_gather_along_first_dim`
```
L114  [HANG] _gather_along_first_dim ENTER
L125  [HANG] _gather_along_first_dim alloc
L129  [HANG] _gather_along_first_dim contigu
L132  [HANG] _gather_along_first_dim ag ENTER
L135  [HANG] _gather_along_first_dim ag EXIT   ← all_gather_into_tensor 返回
L143  [HANG] _gather_along_first_dim EXIT
```

### token_dispatcher.py — MC2 dispatch/combine
```
L240  [HANG] moe dispatch ENTER
L246  [HANG] moe dispatch EXIT               ← ← ← 当前卡点！最后打印的日志
           ↓
      torch.npu.synchronize()  ← 卡在这里，等 MC2 fused kernel 的 HCCL all2all 完成
           ↓
L253  [HANG] moe dispatch post_sync          ← 没打印
L359  [HANG] moe MC2 combine ENTER
L364  [HANG] moe MC2 combine EXIT
L368  [HANG] moe MC2 combine post_sync
```

### ascend_forward_context.py — MoE 通信类型选择
```
[HANG] select_moe_comm_method #N: rank=X num_tokens=Y capacity=Z soc=A3
[HANG] select_moe_comm_method #N: rank=X result=R (0=AG 1=MC2 2=A2A 3=F_MC2)
```

---

## 五、HCCL 底层关键发现

### `blockingWait_=false`（默认）
`ProcessGroupHCCL.cpp` 中 `synchronizeInternal`:
- `blockingWait_` 默认为 `false`（只有 `HCCL_BLOCKING_WAIT=1` 才为 `true`）
- `work->wait()` 只创建计算流对 HCCL 事件的依赖，**不阻塞等待**
- 因此 `all_gather_into_tensor(async_op=False)` 返回时 all_gather 可能还没完成

### `torch.npu.synchronize()` 的局限性
- 调用 `aclrtSynchronizeDevice`，同步 ACL 管理的流
- **不能**保证 HCCL 通信器内部状态被清空
- MC2 fused kernel 在 CANN 内部提交 HCCL 操作，这些操作的状态不在 ACL 流追踪范围

### canary all_reduce 通过 ≠ 通信器空闲
- `dist.all_reduce` 同样只在 `blockingWait_=false` 时设流依赖就返回
- canary "通过"只说明操作被提交，不说明通信器空闲

---

## 六、当前根因分析（最新）

### 卡点：MC2 dispatch 后 `torch.npu.synchronize()`

最后日志 `[HANG] moe dispatch EXIT` → `npu_moe_distribute_dispatch_v2` kernel 返回 → `torch.npu.synchronize()` (L252) 卡住。

### 根因：MC2 使用跨 DP 的 HCCL 通信器

**MC2 通信器包含所有 8 个云 rank（DP0 + DP1）：**

`TokenDispatcherWithMC2.__init__` 中：
```python
device_group = get_mc2_group().device_group
local_rank = torch.distributed.get_rank(group=device_group)
backend = device_group._get_backend(torch.device("npu"))
self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
```

`get_mc2_group()` 在 `parallel_state.py` 中与 EP group 用相同 rank 列表初始化（`[ep_edge_ranks, ep_cloud_ranks]`），因此通信器包含**所有 8 个云 rank**。

`npu_moe_distribute_dispatch_v2` 通过 `moe_all_to_all_group_name` 引用这个通信器，内部提交 HCCL all2all——该 all2all 需要通信器上**所有 8 个 rank** 都提交匹配的操作才能完成。

### 死锁机制

```
DP0 云 rank:  n pu_moe_distribute_dispatch_v2 → HCCL all2all (8-rank)
               → 返回，torch.npu.synchronize() 等 all2all 完成
               → ❌ 等待 DP1 的 rank 也提交 all2all

DP1 云 rank:  还没执行到这个 MoE 层（与前 4 次 DP 同步有微小时间差）
               → ❌ 没提交匹配的 all2all
               → DP0 永远等不到 8 卡凑齐
```

### 前 4 次成功、第 5 次失败的原因

前 4 次 run 中 DP0 和 DP1 恰好同步到达同一 MoE 层，all2all 需要的 8 个 rank 都在。第 5 次某个微小执行时间差（可能是 scheduling order、cache effect、或随机性）导致 DP1 稍慢，DP0 先到并等待。

### A2-A2 不卡的原因

A2-A2 配置下 `select_moe_comm_method` 选择 ALLGATHER fallback（不走 MC2），不会调用 `npu_moe_distribute_dispatch_v2`，不存在跨 DP 通信器的问题。

---

## 七、修改的文件清单

| 文件 | 修改状态 |
|------|---------|
| `vllm-ascend/ops/fused_moe/token_dispatcher.py` | MC2 sync + ALLTOALL sync + 诊断日志 + canary |
| `vllm-ascend/ops/fused_moe/comm_utils.py` | `_gather_along_first_dim` 步骤日志 |
| `vllm-ascend/ascend_forward_context.py` | `select_moe_comm_method` 选择日志 |
| `vllm-ascend/worker/model_runner_v1.py` | 边侧 NaN 检测日志（已加） |

---

## 八、下一步修复方向

### 方向 1：MC2 使用 per-DP 通信器（推荐）
修改 `TokenDispatcherWithMC2.__init__`，不再使用跨 DP 的 `get_mc2_group().device_group`，而是根据 DP rank 创建 per-DP 的子通信器（仅包含同一 DP 内的 4 个云 rank）。

**关键修改点**：`token_dispatcher.py` L108-112
```python
# 当前：跨 DP 的 MC2 group（8-rank）
device_group = get_mc2_group().device_group
local_rank = torch.distributed.get_rank(group=device_group)
backend = device_group._get_backend(torch.device("npu"))
self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
```

**改为**：根据 DP rank 拆分，只用本 DP 内的 4 个 rank 创建通信器。

### 方向 2：以同步方式确保 DP 同步
在 MC2 调用前加一次小规模的 barrier（仅本 DP 内 4 个 rank），确保同一 DP 内同步后再执行 MC2。但不能解决跨 DP 不同步的问题。

### 方向 3：修改 MC2 group 初始化
在 `parallel_state.py` `init_mc2_group` 中，按 DP 拆分创建 per-DP 的 MC2 group（而非全部边云 rank）。

---

## 九、Git 信息
- **仓库**：`git@github.com:wangwei-abcde/vllm-ascend-pdmix.git`
- **分支**：`A2_A3_debug`
- **最新 commit**：`65af2f02` - "fix(MoE): remove barriers that caused HCCL deadlock on cross-DP EP group"
