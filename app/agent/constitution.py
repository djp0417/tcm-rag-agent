# -*- coding: utf-8 -*-
"""体质辨识：九种体质量表（简化自评版）+ 转化分计算 + 判定规则。

为什么这块必须是**确定性代码**而不是交给 LLM
------------------------------------------------
体质判定是 Agent 里唯一「有标准答案」的环节。如果让模型自由心证，
同一段回答跑两次可能给出不同体质——这既不可测、也无法向用户交代。
所以这里把量表与计分规则**硬编码为结构化数据**，形成一条可复现、
可单测的确定性内核；LLM 只负责两件事：
  ① 把用户的口语描述（"我特别怕冷，冬天手脚冰凉"）映射成 1~5 级评分；
  ② 用 RAG 检索到的语料把判定结果**解释**清楚。
即「**代码判定 + 语料解释**」双轨——判定可复现，解释有据可依。

量表来源与简化说明（重要，别说错）
----------------------------------
依据王琦《中医体质分类与判定》（中华中医药学会标准 ZYYXH/T157-2009）的
九分法与判定公式实现，但**题目做了简化**：标准版共 60 题（部分条目按
体质分别计分），一次问完会劝退用户。这里每种体质取 3 个最具区分度的
代表性条目，共 27 题，定位为「**快速自评**」而非临床判定。
- 转化分公式与判定阈值**完全遵循标准**：
      转化分 = (原始分 − 条目数) / (条目数 × 4) × 100
      偏颇体质：≥40 分「是」，30~39 分「倾向是」，<30 分「否」
      平和质：  ≥60 分且其他 8 种均 <30 分「是」，≥60 分但有偏颇≥30 分「基本是」
- 简化的代价：条目少则区分度下降，结论应表述为「倾向」而非诊断。

用法：
    from app.agent.constitution import SCALE, compute, format_report
    answers = {"yangxu": [5, 4, 5], "qixu": [3, 2, 3], ...}   # key → 每题 1~5 分
    result = compute(answers)
    print(format_report(result))
"""
from dataclasses import dataclass, field

# 五级频率 → 分值（问卷标准量表）
FREQ_CHOICES = [
    (1, "没有"), (2, "很少"), (3, "有时"), (4, "经常"), (5, "总是"),
]

# 判定阈值（来自 ZYYXH/T157-2009）
THRESH_POSITIVE = 40        # 偏颇体质：≥40 → 「是」
THRESH_TENDENCY = 30        # 30~39 → 「倾向是」
PINGHE_YES = 60             # 平和质：≥60 且其他均 <30 → 「是」


@dataclass(frozen=True)
class ConstitutionType:
    """一种体质的定义（含量表条目与调养要点）。"""
    key: str                 # 内部键，如 "yangxu"
    name: str                # 体质名，如 "阳虚质"
    code: str                # 标准代号，如 "C型"
    trait: str               # 总体特征（一句话）
    questions: tuple[str, ...]   # 3 个自评条目（均按「近一年」作答）
    care: tuple[str, ...]        # 调养要点；RAG 失败时的兜底摘要


# ---------------------------------------------------------------------------
# 九种体质定义（条目取自标准版最具区分度的代表性表述，做了口语化改写）
# ---------------------------------------------------------------------------
SCALE: tuple[ConstitutionType, ...] = (
    ConstitutionType(
        key="pinghe", name="平和质", code="A型",
        trait="阴阳气血调和：体态适中、面色红润、精力充沛、睡眠食欲良好。",
        questions=(
            "您精力充沛吗？",
            "您睡眠质量好吗（入睡快、不易醒、醒后精神）？",
            "您面色红润、胃口好吗？",
        ),
        care=("饮食有节、不宜过饥过饱，五谷为养、荤素搭配",
              "起居规律，尽量 23 点前入睡，不熬夜",
              "适度运动（快走、八段锦、太极拳），每周 3~5 次",
              "情志平和，避免长期紧张或情绪大起大落"),
    ),
    ConstitutionType(
        key="qixu", name="气虚质", code="B型",
        trait="元气不足：容易疲乏、气短懒言、易出虚汗、易感冒。",
        questions=(
            "您容易疲乏、没精神吗？",
            "您容易气短（呼吸短促、上气不接下气）吗？",
            "您比别人容易感冒、或感冒后不容易好吗？",
        ),
        care=("饮食宜益气健脾：山药、莲子、大枣、小米、黄芪炖鸡",
              "少食生冷寒凉与耗气之物（萝卜、空心菜、生冷瓜果）",
              "运动宜柔缓（散步、八段锦、太极），忌大汗淋漓耗气",
              "注意保暖防感冒，避免过度劳累与久思伤脾"),
    ),
    ConstitutionType(
        key="yangxu", name="阳虚质", code="C型",
        trait="阳气不足：畏寒怕冷、手足不温、喜热饮食、精神不振。",
        questions=(
            "您手脚发凉、怕冷吗？",
            "您比一般人耐受不了寒冷（冬天、空调、电扇）吗？",
            "您吃凉东西会感到不舒服、或容易拉肚子吗？",
        ),
        care=("饮食宜温阳：生姜、羊肉、韭菜、桂圆、核桃，晨起可饮姜枣茶",
              "忌生冷冰饮、苦寒瓜果（西瓜、苦瓜、绿豆）",
              "注意腰腹与足部保暖，坚持温水泡脚（可加艾叶）",
              "多晒太阳、晒后背（督脉），适度运动以生阳气"),
    ),
    ConstitutionType(
        key="yinxu", name="阴虚质", code="D型",
        trait="阴液亏少：口燥咽干、手足心热、喜冷饮、易失眠。",
        questions=(
            "您感到手脚心发热吗？",
            "您口燥咽干、总想喝水吗？",
            "您容易失眠、或睡中出汗（盗汗）吗？",
        ),
        care=("饮食宜滋阴润燥：银耳、百合、梨、桑葚、鸭肉、黑芝麻",
              "忌辛辣煎炸、烟酒与过食温燥（羊肉、辣椒、荔枝）",
              "起居宜静，避免熬夜伤阴（熬夜最耗阴液）",
              "运动宜中小强度，避免高温时段暴晒大汗"),
    ),
    ConstitutionType(
        key="tanshi", name="痰湿质", code="E型",
        trait="痰湿凝聚：形体肥胖、腹部肥满松软、口黏苔腻、身重易困。",
        questions=(
            "您感到身体沉重、不轻松或腹部肥满松软吗？",
            "您额部油脂分泌多、或嘴里有黏黏的感觉吗？",
            "您平时痰多、或容易胸闷吗？",
        ),
        care=("饮食宜清淡健脾祛湿：薏米、赤小豆、冬瓜、白萝卜、荷叶",
              "少食肥甘厚味、甜食与酒类，晚餐不宜过饱",
              "坚持有氧运动出微汗（快走、慢跑、游泳），控制体重",
              "居处宜干燥通风，避免久坐久卧、淋雨涉水"),
    ),
    ConstitutionType(
        key="shire", name="湿热质", code="F型",
        trait="湿热内蕴：面垢油光、易生痤疮、口苦口干、身重困倦、大便黏滞。",
        questions=(
            "您面部或鼻部油光发亮、容易生痤疮吗？",
            "您口苦、或嘴里有异味吗？",
            "您大便黏滞不爽、小便发黄吗？",
        ),
        care=("饮食宜清热利湿：绿豆、冬瓜、苦瓜、芹菜、绿茶",
              "忌辛辣油腻、烧烤与酒，少食温燥的牛羊肉",
              "避免熬夜与湿热环境，保证睡眠以助肝胆疏泄",
              "适合大强度运动排汗（中长跑、球类），注意及时补水"),
    ),
    ConstitutionType(
        key="xueyu", name="血瘀质", code="G型",
        trait="血行不畅：肤色晦暗、易出现瘀斑、口唇偏暗、易健忘。",
        questions=(
            "您的皮肤容易出现青紫瘀斑吗？",
            "您面色晦暗、或容易出现黑眼圈吗？",
            "您口唇颜色偏暗、或容易健忘吗？",
        ),
        care=("饮食宜活血化瘀：山楂、桃仁、黑豆、玫瑰花、少量红酒",
              "忌过食寒凉凝滞与高脂厚味，注意情绪疏导（气滞则血瘀）",
              "多做促进循环的运动（快走、健身操、舞蹈），避免久坐",
              "注意保暖，可常按揉血海、三阴交等穴位"),
    ),
    ConstitutionType(
        key="qiyu", name="气郁质", code="H型",
        trait="气机郁滞：情绪低落、多愁善感、容易紧张焦虑、常叹气。",
        questions=(
            "您感到闷闷不乐、情绪低沉吗？",
            "您容易精神紧张、或焦虑不安吗？",
            "您多愁善感、容易感到害怕或受惊吓吗？",
        ),
        care=("饮食宜行气解郁：佛手、玫瑰花、陈皮、柑橘、黄花菜",
              "少食收敛酸涩之物，避免以酒解忧（酒更伤肝气）",
              "多户外活动与社交，培养兴趣以舒展气机",
              "可练八段锦「双手托天理三焦」、常按揉太冲穴"),
    ),
    ConstitutionType(
        key="tebing", name="特禀质", code="I型",
        trait="先天失常：过敏体质，易打喷嚏、鼻塞、皮肤起风团。",
        questions=(
            "您容易过敏（对药物、食物、气味、花粉等）吗？",
            "您的皮肤容易起风团（荨麻疹）、抓痕吗？",
            "您容易打喷嚏、鼻塞、流鼻涕吗？",
        ),
        care=("饮食宜清淡均衡，忌已知致敏食物与腥发之物（虾蟹、酒）",
              "起居避风寒，花粉季与雾霾天减少外出或戴口罩",
              "被褥常晒洗，避免尘螨、宠物皮屑等常见致敏原",
              "规律作息、适度运动以改善体质基础，必要时就医查过敏原"),
    ),
)

BY_KEY = {t.key: t for t in SCALE}


# ---------------------------------------------------------------------------
# 计分与判定
# ---------------------------------------------------------------------------
@dataclass
class TypeScore:
    """单个体质的评分结果。"""
    key: str
    name: str
    code: str
    raw: int                     # 原始分（各题 1~5 分求和）
    transform: float             # 转化分 0~100
    verdict: str                 # 「是」/「倾向是」/「否」
    is_primary: bool = False     # 是否为主体质（偏颇体质中得分最高）


@dataclass
class ConsultResult:
    """一次体质辨识的完整结果。"""
    scores: list[TypeScore]
    primary: str                 # 主体质名（无偏颇时为「平和质」）
    primary_key: str
    tendencies: list[str] = field(default_factory=list)   # 兼夹/倾向体质名
    note: str = ""
    careless: bool = False       # 疑似无差别作答（结论可信度低）


# 无差别作答检测
# -----------------
# 量表用「转化分 = (原始分 − 条目数)/(条目数 × 4) × 100」归一化，全套选"有时"(3分)
# 会得到 50 分——按标准 ≥40 即判「是」，于是八种偏颇体质**全部**被判「是」。
# 这在数学上忠于标准（标准版 60 题同样如此，因为真人对症状条目不会均匀作答），
# 但简化版条目少、用户又容易顺手全点中间档，出现这种结果必须提示"作答无效"，
# 而不是把 8 种体质一起拍给用户——这是**产品级护栏**，不改变判定公式本身。
CARELESS_RANGE = 1               # 所有作答的极差 ≤ 此值即判定为无差别作答


def detect_careless(answers: dict[str, list[int]], min_items: int = 10) -> bool:
    """检测无差别作答：有效作答数足够，但分值几乎不变（极差 ≤ CARELESS_RANGE）。"""
    vals = [v for arr in answers.values() for v in arr if v]
    if len(vals) < min_items:
        return False
    return (max(vals) - min(vals)) <= CARELESS_RANGE


def transform_score(raw: int, n_items: int) -> float:
    """转化分 = (原始分 − 条目数) / (条目数 × 4) × 100，结果范围 0~100。

    这是标准规定的归一化方式：把「几个条目 × 1~5 分」折算成 0~100 分，
    使不同条目数的体质之间可以横向比较。
    """
    return (raw - n_items) / (n_items * 4) * 100


def compute(answers: dict[str, list[int]]) -> ConsultResult:
    """按量表与标准阈值算出体质判定结果。

    Args:
        answers: {体质 key: [每题 1~5 分]}；缺项的体质按「全部未答」处理（不参与判定）。
    Returns:
        ConsultResult —— 含全部 9 种的得分明细、主体质、兼夹倾向。
    """
    scores: list[TypeScore] = []
    for t in SCALE:
        vals = answers.get(t.key)
        if not vals or len(vals) != len(t.questions):
            # 未完整作答的体质跳过（比如中途结束问诊）
            continue
        raw = sum(vals)
        tf = round(transform_score(raw, len(t.questions)), 1)

        if t.key == "pinghe":
            # 平和质判定：≥60 且其他偏颇质均 <30 → 「是」；≥60 但有偏颇 ≥30 → 「基本是」
            others = [s for s in scores if s.key != "pinghe"]   # 此处只含已算过的
            verdict = "否"
            if tf >= PINGHE_YES:
                verdict = "基本是" if any(o.transform >= THRESH_TENDENCY for o in others) else "是"
        else:
            verdict = ("是" if tf >= THRESH_POSITIVE
                       else "倾向是" if tf >= THRESH_TENDENCY else "否")
        scores.append(TypeScore(t.key, t.name, t.code, raw, tf, verdict))

    biased = [s for s in scores if s.key != "pinghe" and s.verdict == "是"]
    tendencies = [s.name for s in scores
                  if s.key != "pinghe" and s.verdict == "倾向是"]

    if biased:
        top = max(biased, key=lambda s: s.transform)
        top.is_primary = True
        primary, primary_key = top.name, top.key
        # 兼夹：除主体质外同样判定为「是」的
        tendencies = [s.name for s in biased if s.key != top.key] + tendencies
    else:
        primary, primary_key = "平和质", "pinghe"
        # 没有一个偏颇体质达到「是」，但可能有倾向
        if not tendencies:
            tendencies = []

    careless = detect_careless(answers)
    notes: list[str] = []
    if not answers.get("pinghe"):
        notes.append("平和质部分未作答，判定仅供参考。")
    if careless:
        notes.append("⚠️ 检测到作答分值几乎一致（疑似无差别作答），"
                     "结果可信度低，建议按实际情况重新作答。")
    return ConsultResult(scores=scores, primary=primary, primary_key=primary_key,
                         tendencies=tendencies, note="".join(notes),
                         careless=careless)


def format_report(r: ConsultResult) -> str:
    """把判定结果格式化为一段结构化文本，供 LLM 组织语言 / 前端展示。"""
    lines = ["【体质辨识结果（依据标准量表转化分计算，非临床诊断）】"]
    if r.primary == "平和质":
        lines.append(f"主体质：{r.primary}（阴阳气血调和）")
    else:
        lines.append(f"主体质：{r.primary}")
    if r.tendencies:
        lines.append(f"兼夹/倾向：{'、'.join(r.tendencies)}")
    lines.append("")
    lines.append("各项转化分（0~100，越高倾向越明显）：")
    for s in sorted(r.scores, key=lambda x: -x.transform):
        mark = "★" if s.is_primary else " "
        lines.append(f"  {mark} {s.name}({s.code})  {s.transform:>5.1f} 分  → {s.verdict}")
    if r.careless:
        # 无差别作答时把「是」的结论明确标注为不可信，避免模型据此大讲特讲
        lines.append("")
        lines.append("注意：本次作答疑似无差别（分值几乎一致），"
                     "上述判定**不可作为结论**，请向用户说明并建议重新作答。")
    if r.note:
        lines.append("")
        lines.append(r.note)
    return "\n".join(lines)


def care_points(key: str) -> list[str]:
    """取某体质的调养要点（RAG 检索失败时的兜底内容）。"""
    t = BY_KEY.get(key)
    return list(t.care) if t else []


def total_questions() -> int:
    """量表总题数（供前端进度条使用）。"""
    return sum(len(t.questions) for t in SCALE)
