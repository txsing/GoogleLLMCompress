# 第一组命令，输出写入 log1.txt
python train_enwik_torch.py -e 1 -b 128; python compress_enwik_torch.py -m params_epoch_1.pth > log1.txt 2>&1 &

# 第二组命令，输出写入 log2.txt
python train_enwik_torch.py -e 2 -b 128; python compress_enwik_torch.py -m params_epoch_2.pth > log2.txt 2>&1 &

# 第三组命令，输出写入 log3.txt
python train_enwik_torch.py -e 3 -b 128; python compress_enwik_torch.py -m params_epoch_3.pth > log3.txt 2>&1 &

wait  # 等待所有任务完成
echo "所有任务执行完毕，日志已分别写入 log1.txt、log2.txt、log3.txt"
