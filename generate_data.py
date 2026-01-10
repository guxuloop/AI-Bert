import json
import random
import os

# ==========================================
# 1. 基础词库 (扩充这个列表可以增加数据多样性)
# ==========================================

TIMES = [
    "明天", "后天", "下周一", "下周五", "周末", "今天下午", 
    "晚上8点", "上午10点", "下午三点", "早上9:00", "5分钟后", 
    "2024年1月1日", "春节期间", "这周日", "每天早上"
]

LOCATIONS = [
    "会议室", "办公室", "家里", "星巴克", "机场", 
    "上海", "北京", "302室", "楼下食堂", "客户公司", 
    "大厅", "前台", "学校", "健身房", "公园", "火锅店", 
    "电影院", "商场", "医院", "图书馆"
]

PEOPLE = [
    "男朋友", "女朋友", "老王", "同事", "爸妈", "朋友", 
    "客户", "老板", "小李", "张总", "我妈", "室友"
]

CONTENTS = [
    "开会", "写代码", "面试", "吃饭", "看电影", 
    "见客户", "讨论需求", "发周报", "取快递", 
    "体检", "做PPT", "提交代码", "修Bug", "打球",
    "吃火锅", "喝咖啡", "聚餐", "约会", "逛街"
]

ALARM_ACTIONS = [
    "起床", "吃药", "喝水", "关火", "跑步", 
    "睡觉", "出发", "休息", "喂猫", "抢票"
]

OTHERS = [
    "今天天气怎么样", "给我讲个笑话", "你是谁", 
    "播放音乐", "打开空调", "现在的股价是多少", 
    "不仅如此", "然而", "这就很尴尬了", "哈哈哈哈",
    "推荐一部电影", "搜索一下附近的美食"
]

# ==========================================
# 2. 生成函数
# ==========================================

def get_bio_tags(text, entity_type=None):
    """
    将文本转换为 BIO 标签序列。
    例如: "明天", "TIME" -> ["B-TIME", "I-TIME"]
    如果是 None -> ["O", "O"]
    """
    if not entity_type:
        return ["O"] * len(text)
    
    tags = []
    for i, char in enumerate(text):
        if i == 0:
            tags.append(f"B-{entity_type}")
        else:
            tags.append(f"I-{entity_type}")
    return tags

def generate_schedule_sample():
    """生成一条[日程]数据，同时返回 NER 标注"""
    # 随机选择模板结构
    structure = random.choice([
        ["T", "在", "L", "DO", "C"],            # 明天 在 会议室 DO 开会
        ["T", "去", "L", "DO", "C"],            # 下周一 去 上海 DO 见客户
        ["T", "DO", "C"],                       # 晚上8点 DO 写代码
        ["C", "T"],                             # 面试 下周五 (倒装)
        ["帮忙安排", "T", "的", "C"],            # 帮忙安排 明天 的 会议
        ["T", "跟", "P", "一起", "C"],           # 明天 跟 男朋友 一起 吃饭 (P应标为O)
        ["T", "和", "P", "去", "L", "C"],        # 明天 和 同事 去 食堂 吃饭
        ["T", "约了", "P", "在", "L", "C"]       # 明天 约了 客户 在 公司 开会
    ])

    chars = []
    ner_tags = []
    
    t_val = random.choice(TIMES)
    l_val = random.choice(LOCATIONS)
    c_val = random.choice(CONTENTS)
    p_val = random.choice(PEOPLE)

    for item in structure:
        if item == "T":
            chars.extend(list(t_val))
            ner_tags.extend(get_bio_tags(t_val, "TIME"))
        elif item == "L":
            chars.extend(list(l_val))
            ner_tags.extend(get_bio_tags(l_val, "LOC"))
        elif item == "C":
            chars.extend(list(c_val))
            ner_tags.extend(get_bio_tags(c_val, "CONTENT"))
        elif item == "P":
            # 人物不属于我们需要提取的实体 (O)
            chars.extend(list(p_val))
            ner_tags.extend(get_bio_tags(p_val, None)) 
        elif item == "DO": 
            # 随机插入一些连接动词，标记为 O
            s = random.choice(["", "准备", "要"])
            chars.extend(list(s))
            ner_tags.extend(get_bio_tags(s, None))
        else:
            # 固定连接词
            chars.extend(list(item))
            ner_tags.extend(get_bio_tags(item, None))

    full_text = "".join(chars)
    return {
        "text": full_text,
        "label": 0, # 日程
        "ner_tokens": chars,
        "ner_tags": ner_tags
    }

def generate_alarm_sample():
    """生成一条[闹钟]数据"""
    # 闹钟通常只关心 TIME，有时候关心 CONTENT(作为备注)
    structure = random.choice([
        ["T", "叫我", "A"],             # 早上8点 叫我 起床
        ["定一个", "T", "的闹钟"],      # 定一个 明天 的闹钟
        ["提醒我", "T", "DO", "A"],     # 提醒我 5分钟后 DO 喝水
    ])

    chars = []
    ner_tags = []
    
    t_val = random.choice(TIMES)
    a_val = random.choice(ALARM_ACTIONS) # 闹钟的内容通常比较简单

    for item in structure:
        if item == "T":
            chars.extend(list(t_val))
            ner_tags.extend(get_bio_tags(t_val, "TIME"))
        elif item == "A":
            # 这里的 A 可以看作 CONTENT 也可以忽略，暂且标记为 CONTENT
            chars.extend(list(a_val))
            ner_tags.extend(get_bio_tags(a_val, "CONTENT"))
        elif item == "DO":
            s = random.choice(["", "去"])
            chars.extend(list(s))
            ner_tags.extend(get_bio_tags(s, None))
        else:
            chars.extend(list(item))
            ner_tags.extend(get_bio_tags(item, None))

    full_text = "".join(chars)
    return {
        "text": full_text,
        "label": 1, # 闹钟
        "ner_tokens": chars,
        "ner_tags": ner_tags
    }

def generate_other_sample():
    """生成一条[其他]数据 (负样本)"""
    text = random.choice(OTHERS)
    # 随机加点标点或者前后缀
    if random.random() > 0.5:
        text += "吗"
    
    chars = list(text)
    ner_tags = ["O"] * len(chars)

    return {
        "text": text,
        "label": 2, # 其他
        "ner_tokens": chars,
        "ner_tags": ner_tags
    }

# ==========================================
# 3. 主程序
# ==========================================

def main():
    CLS_COUNT = 10000  # 生成10000条分类数据
    NER_COUNT = 10000  # 生成10000条NER数据
    
    os.makedirs("data", exist_ok=True)
    
    samples = []
    
    # 生成数据
    for _ in range(int(CLS_COUNT * 0.4)): # 40% 日程
        samples.append(generate_schedule_sample())
    
    for _ in range(int(CLS_COUNT * 0.4)): # 40% 闹钟
        samples.append(generate_alarm_sample())
        
    for _ in range(int(CLS_COUNT * 0.2)): # 20% 其他
        samples.append(generate_other_sample())

    random.shuffle(samples)

    # 写入分类数据集
    print(f"Generating {len(samples)} classification samples to data/classifier_train_large.jsonl...")
    with open("data/classifier_train_large.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            item = {"text": s["text"], "label": s["label"]}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 写入 NER 数据集 (只取有实体的数据，或者包含部分 O 样本)
    # 这里我们直接复用生成的数据，因为它们都带了 tagging
    print(f"Generating {len(samples)} NER samples to data/ner_train_large.jsonl...")
    with open("data/ner_train_large.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            # HuggingFace datasets 需要 'tokens' 和 'ner_tags' 字段
            item = {
                "tokens": s["ner_tokens"],
                "ner_tags": s["ner_tags"]
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print("Done! New datasets are ready.")

if __name__ == "__main__":
    main()
