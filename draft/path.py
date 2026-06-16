import os

def print_tree(root_dir):
    for root, dirs, files in os.walk(root_dir):
        level = root.replace(root_dir, '').count(os.sep)
        indent = '│   ' * level
        print(f"{indent}├── {os.path.basename(root)}/")
        sub_indent = '│   ' * (level + 1)
        for f in files:
            print(f"{sub_indent}├── {f}")

if __name__ == "__main__":
    path = "../"   # ← 换成你要看的路径
    print_tree(path)
