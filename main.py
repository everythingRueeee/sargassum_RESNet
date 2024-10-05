from scripts.train import train_model
from scripts.validate import validate_model
from scripts.test import test_model


def main():
    while True:
        operation = input("请输入要执行的操作 (train/val/test): ").strip().lower()

        if operation == "train":
            print("开始训练模型...")
            train_model()
            break
        elif operation == "val":
            print("开始验证模型...")
            validate_model()  # 独立运行验证
            break
        elif operation == "test":
            print("开始测试模型...")
            test_model()
            break
        else:
            print("无效的操作，请选择 'train', 'val' 或 'test'.")


if __name__ == "__main__":
    main()
