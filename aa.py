import json

# 读取文本文件内容
txt_file_path = "./test_true_extraction.txt"
with open(txt_file_path, "r") as txt_file:
    text_data = txt_file.read()

# 创建包含文本数据的字典
data = {"text": text_data}

# 将字典转换为 JSON 格式
json_data = json.dumps(data, indent=4)

# 写入 JSON 数据到新文件
json_file_path = "./output.json"
with open(json_file_path, "w") as json_file:
    json_file.write(json_data)

print("转换完成，JSON 文件已保存为:", json_file_path)

