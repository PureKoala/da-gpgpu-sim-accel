#!/usr/bin/env python3
"""
测试DAT模型的维度正确性
"""
import torch
import sys
import os
import pandas as pd
sys.path.insert(0, '/home/koala/gpgpu-sim_distribution/DAT/py')
from DAT import DAT_base_for_sim

def test_dat_dimensions():
    """测试DAT模型各个阶段的维度"""
    print("=" * 60)
    print("测试 DAT-base 模型维度")
    print("=" * 60)
    
    # 创建模型
    model = DAT_base_for_sim(
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dims=[128, 256, 512, 1024],
        num_heads=[4, 8, 16, 32],
        num_blocks=[1, 1, 9, 1],
        offset_groups=[1, 2, 4, 8],
        grid_size=7,
        drop_rate=0.,
        drop_path_rate=0.1
    )
    model.eval()
    
    # 创建测试输入
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    print(f"\n输入形状: {x.shape}")
    
    # 前向传播
    try:
        with torch.no_grad():
            output = model(x)
        print(f"✓ 输出形状: {output.shape}")
        print(f"✓ 期望形状: ({batch_size}, 1000)")
        
        if output.shape == (batch_size, 1000):
            print("\n" + "=" * 60)
            print("✓✓✓ 所有维度测试通过！")
            print("=" * 60)
            return True
        else:
            print("\n" + "=" * 60)
            print("✗✗✗ 输出维度不匹配！")
            print("=" * 60)
            return False
            
    except Exception as e:
        print(f"\n✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_deformable_attention():
    """测试DeformableAttention模块"""
    print("\n" + "=" * 60)
    print("测试 DeformableAttention 模块")
    print("=" * 60)
    
    from DAT import DeformableAttention
    
    # 参数
    B, H, W = 2, 56, 56
    N = H * W  # 3136
    C = 128
    M = 4  # num_heads
    G = 1  # offset_groups
    
    # 创建模块
    attn = DeformableAttention(
        dim=C,
        num_heads=M,
        grid_size=7,
        offset_groups=G,
        use_rpb=True
    )
    attn.eval()
    
    # 创建输入
    x = torch.randn(B, N, C)
    hw_shape = (H, W)
    
    print(f"\n输入: x={x.shape}, hw_shape={hw_shape}")
    
    try:
        with torch.no_grad():
            output = attn(x, hw_shape)
        print(f"✓ 输出形状: {output.shape}")
        print(f"✓ 期望形状: ({B}, {N}, {C})")
        
        if output.shape == (B, N, C):
            print("\n✓✓✓ DeformableAttention 测试通过！")
            return True
        else:
            print("\n✗✗✗ 输出维度不匹配！")
            return False
            
    except Exception as e:
        print(f"\n✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_stages():
    """测试每个stage的输出维度"""
    print("\n" + "=" * 60)
    print("测试各Stage输出维度")
    print("=" * 60)
    
    model = DAT_base_for_sim()
    model.eval()
    
    B = 2
    x = torch.randn(B, 3, 224, 224)
    
    expected_shapes = [
        (B, 3136, 128),   # Stage1: 56x56
        (B, 784, 256),    # Stage2: 28x28
        (B, 196, 512),    # Stage3: 14x14
        (B, 49, 1024),    # Stage4: 7x7
    ]
    
    try:
        with torch.no_grad():
            # Patch Embed
            x, hw_shape = model.patch_embed(x)
            print(f"\nPatchEmbed: {x.shape}, hw={hw_shape}")
            
            stage_idx = 0
            for i in range(0, len(model.stages), 2):
                # Stage blocks
                blocks = model.stages[i]
                for block in blocks:
                    x = block(x, hw_shape)
                
                print(f"Stage{stage_idx+1}: {x.shape}, hw={hw_shape}")
                
                if x.shape != expected_shapes[stage_idx]:
                    print(f"  ✗ 期望: {expected_shapes[stage_idx]}")
                    return False
                else:
                    print(f"  ✓ 维度正确")
                
                # Downsample
                if i + 1 < len(model.stages):
                    downsample = model.stages[i + 1]
                    x, hw_shape = downsample(x, hw_shape)
                
                stage_idx += 1
            
            print("\n✓✓✓ 所有Stage维度正确！")
            return True
            
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_profiling(idx=0, use_cuda=False):
    """
    测试模型并进行性能分析
    
    参数:
        idx: 测试索引，0表示进行profiling
        use_cuda: 是否使用CUDA
    """
    print("\n" + "=" * 60)
    print("性能分析测试")
    print("=" * 60)
    
    model = DAT_base_for_sim()
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    batch_size = 2
    lq = torch.randn(batch_size, 3, 224, 224).to(device)
    
    print(f"设备: {device}")
    
    with torch.no_grad():
        if idx == 0:
            log_dir = "./log_inference"
            os.makedirs(log_dir, exist_ok=True)
            
            # 预热
            print("预热运行...")
            for i in range(3):
                if use_cuda and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        _ = model(lq)
                else:
                    _ = model(lq)
            
            activities = [torch.profiler.ProfilerActivity.CPU]
            if use_cuda and torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            
            print("开始性能分析...")
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                with_stack=True,
                profile_memory=True
            ) as prof:
                if use_cuda and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        output = model(lq)
                else:
                    output = model(lq)
            
            # 打印摘要
            print("\n--- Profiler Summary (Top 20) ---")
            sort_key = "cuda_time_total" if use_cuda and torch.cuda.is_available() else "cpu_time_total"
            print(prof.key_averages().table(sort_by=sort_key, row_limit=20))
            
            # 导出trace
            prof.export_chrome_trace(os.path.join(log_dir, "trace_inference.json"))
            
            # 收集数据
            chronological_data = []
            if use_cuda and torch.cuda.is_available():
                for event in prof.events():
                    if event.device_type == torch.autograd.DeviceType.CUDA:
                        chronological_data.append({
                            "Operator Name": event.name,
                            "Start Time (us)": event.time_range.start,
                            "Duration (us)": event.cuda_time_total
                        })
            else:
                for event in prof.events():
                    if event.device_type == torch.autograd.DeviceType.CPU:
                        chronological_data.append({
                            "Operator Name": event.name,
                            "Start Time (us)": event.time_range.start,
                            "Duration (us)": event.cpu_time_total
                        })
            
            if not chronological_data:
                print("[Error]: 没有成功分析到操作数据。")
            else:
                df_chronological = pd.DataFrame(chronological_data)
                df_summary = df_chronological.groupby('Operator Name').agg(
                    Total_Duration_us=('Duration (us)', 'sum'),
                    Call_Count=('Operator Name', 'size')
                )
                df_summary_sorted = df_summary.sort_values(by='Total_Duration_us', ascending=False).reset_index()
                
                with pd.ExcelWriter(os.path.join(log_dir, "cuda_profile.xlsx"), engine='openpyxl') as writer:
                    df_chronological.to_excel(writer, sheet_name='Sheet1_Chronological', index=False)
                    df_summary_sorted.to_excel(writer, sheet_name='Sheet2_Summary', index=False)
                
                print(f"\n✓ 结果已导出:")
                print(f"  - JSON: {log_dir}/trace_inference.json")
                print(f"  - Excel: {log_dir}/cuda_profile.xlsx")
        else:
            output = model(lq)
        
        return True

if __name__ == "__main__":
    print("\n开始测试 DAT 模型...\n")
    
    use_cuda = torch.cuda.is_available()
    
    # 运行所有测试
    test1 = test_deformable_attention()
    test2 = test_all_stages()
    test3 = test_dat_dimensions()
    
    # 可选: 运行性能分析 (取消注释下面这行)
    test4 = test_with_profiling(idx=0, use_cuda=use_cuda)
    
    print("\n" + "=" * 60)
    print("测试总结:")
    print("=" * 60)
    print(f"DeformableAttention: {'✓ 通过' if test1 else '✗ 失败'}")
    print(f"各Stage维度测试:    {'✓ 通过' if test2 else '✗ 失败'}")
    print(f"完整模型测试:       {'✓ 通过' if test3 else '✗ 失败'}")
    print("=" * 60)
    
    if test1 and test2 and test3:
        print("\n🎉 所有测试通过！模型实现正确。")
        print("\n提示: 要运行性能分析，请使用:")
        print("  python test_dat_profile.py")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败，请检查模型实现。")
        sys.exit(1)
