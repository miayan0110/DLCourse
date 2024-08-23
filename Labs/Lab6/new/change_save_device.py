from model import ConditionDDPM
import torch

if __name__ == '__main__':
    # 假設模型的權重存儲在這個路徑
    model_path = 'results/ckpt/last_223_cuda0.pt'

    # 加載模型
    device = torch.device('cuda:1')  # 設定目前設備為CPU
    model = ConditionDDPM(3, 24).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # 轉移到另一個設備（例如轉移到GPU）
    new_device = torch.device('cuda:0')
    model.to(new_device)

    # 保存模型到新的設備
    new_model_path = 'results/ckpt/last_223_cuda0.pt'
    torch.save({
            'epoch': checkpoint['epoch'],
            'model': model.state_dict(),
            'optimizer': checkpoint['optimizer']
        }, new_model_path)

    print(f"Model weights saved to {new_model_path} on device {new_device}")