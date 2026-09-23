# -*- coding: utf-8 -*-
"""
面向电商智能客服的中文问句语义相似度度量构建 —— 实验脚本
零第三方依赖（纯 Python），可复现论文全部实验数据。

度量实现：
  M1  Jaccard 相似度（字符 bigram）
  M2  归一化编辑距离相似度（1 - NLev）
  M3  TF-IDF 余弦相似度（字符 3-gram）
  M4  分布词向量余弦（自构建 PPMI 词向量 + 平均池化，无需预训练模型）
  M5  融合度量（字面 + 语义，权重由验证集网格搜索）
评估：阈值扫描 F1、AUC（排序近似）、AP、消融
"""
import json, math, os, itertools, time

# ======================================================================
# 一、电商问句对数据集（手工构建，含等价/不等价标签与 0-1 相似度等级）
#     设计覆盖：同义改写、词序变化、错别字、近义区分、字面重叠但语义无关、
#               长尾冷门、跨品类无关等情形，以考察各度量优劣。
# ======================================================================
DATASET = [
    # (问句A, 问句B, 等价标签1/0, 人工相似度0-1, 情形标签)
    ("怎么退款", "退款流程是什么", 1, 0.90, "同义改写"),
    ("退款怎么操作", "如何申请退款", 1, 0.92, "同义改写"),
    ("我要退货", "退货流程", 1, 0.88, "同义改写"),
    ("快递到哪了", "我的订单物流", 1, 0.85, "同义改写"),
    ("订单到哪了", "查询物流进度", 1, 0.83, "同义改写"),
    ("优惠券怎么用", "优惠券使用规则", 1, 0.90, "同义改写"),
    ("包邮吗", "运费多少", 1, 0.80, "同义改写"),
    ("多久能到", "几天能送达", 1, 0.85, "同义改写"),
    ("支持货到付款吗", "可以货到付款吗", 1, 0.95, "词序变化"),
    ("可以开发票吗", "能开发票吗", 1, 0.96, "词序变化"),
    ("七天无理由退货吗", "支持七天无理由退货", 1, 0.97, "词序变化"),
    ("怎么付款", "如何支付", 1, 0.90, "同义改写"),
    ("会员积分怎么算", "积分规则", 1, 0.82, "同义改写"),
    ("商品是正品吗", "保证正品吗", 1, 0.88, "同义改写"),
    ("有货吗", "库存还有吗", 1, 0.85, "同义改写"),
    ("能不能分期", "支持分期付款吗", 1, 0.88, "同义改写"),
    ("尺寸怎么选", "尺码怎么挑", 1, 0.90, "近义替换"),
    ("颜色有几个", "有哪些颜色可选", 1, 0.88, "同义改写"),
    ("保修多久", "质保期多长", 1, 0.90, "近义替换"),
    ("发货地是哪", "从哪里发货", 1, 0.88, "词序变化"),
    ("退货流程", "退换货政策", 1, 0.80, "近义替换"),
    ("我的订单到哪了", "订单状态查询", 1, 0.85, "同义改写"),
    ("手机有什么优惠", "这款手机有折扣吗", 1, 0.88, "同义改写"),
    ("能便宜点吗", "能优惠吗", 1, 0.85, "近义替换"),
    ("发票怎么开", "怎么开发票", 1, 0.96, "词序变化"),
    ("怎么退换货", "退换货怎么操作", 1, 0.93, "词序变化"),
    ("支持七天无理由", "七天后还能退吗", 0, 0.20, "近义但语义对立"),
    ("退款多久到账", "退款被拒了", 0, 0.15, "字面重叠语义无关"),
    ("包邮吗", "包安装吗", 0, 0.20, "字面重叠语义无关"),
    ("怎么退款", "怎么充值", 0, 0.25, "字面重叠语义无关"),
    ("优惠券怎么用", "优惠券过期了", 0, 0.30, "字面重叠语义无关"),
    ("发货快吗", "退货快吗", 0, 0.25, "字面重叠语义对立"),
    ("会员积分", "会员等级", 0, 0.20, "同域近义但不同义"),
    ("发票怎么开", "发票丢了", 0, 0.30, "字面重叠语义无关"),
    ("手机优惠", "电脑优惠", 0, 0.40, "品类不同"),
    ("篮球多少钱", "足球多少钱", 0, 0.45, "品类不同"),
    ("咖啡豆产地", "咖啡豆保质期", 0, 0.45, "同品不同属性"),
    ("耳机降噪怎么样", "音箱音质怎么样", 0, 0.40, "品类不同"),
    ("退款流程", "投诉流程", 0, 0.35, "字面重叠语义无关"),
    ("包邮吗", "包邮门槛多少", 1, 0.80, "近义追问"),
    ("能开发票吗", "发票抬头怎么改", 1, 0.75, "同主题相关"),
    ("手机内存多大", "手机存储多少G", 1, 0.92, "同义改写"),
    ("笔记本续航多久", "电脑能用几小时", 1, 0.85, "同义改写"),
    ("耳机防水吗", "耳机能沾水吗", 1, 0.88, "同义改写"),
    ("手表能测心率吗", "智能手表心率监测", 1, 0.85, "同义改写"),
    ("扫地机器人吸力多大", "扫地机吸力参数", 1, 0.88, "同义改写"),
    ("篮球是室内的吗", "篮球室内外通用吗", 1, 0.80, "近义追问"),
    ("跑鞋透气吗", "运动鞋透气性", 1, 0.85, "同义改写"),
    ("卫衣起球吗", "卫衣会不会起球", 1, 0.92, "同义改写"),
    ("咖啡豆是深烘吗", "咖啡豆烘焙度", 1, 0.82, "同义改写"),
    ("坚果含糖吗", "每日坚果有糖吗", 1, 0.80, "同义改写"),
    ("帮我推荐手机", "有什么手机推荐", 1, 0.90, "同义改写"),
    ("推荐个礼物", "送什么礼物好", 1, 0.85, "同义改写"),
    ("篮球怎么选", "篮球选购指南", 1, 0.85, "同义改写"),
    ("跑鞋偏码吗", "跑鞋尺码偏大吗", 1, 0.88, "同义改写"),
    ("耳机续航多久", "耳机能用多久", 1, 0.88, "同义改写"),
    ("手表防水吗", "手表能下水吗", 1, 0.88, "同义改写"),
    ("扫地机噪音大吗", "扫地机器人声音大吗", 1, 0.90, "同义改写"),
    ("咖啡豆新鲜吗", "咖啡豆现烘吗", 1, 0.85, "同义改写"),
    ("发票能改吗", "发票能修改吗", 1, 0.95, "同义改写"),
    ("手机有现货吗", "手机什么时候有货", 1, 0.80, "近义追问"),
    ("退货运费谁出", "退货要付运费吗", 1, 0.88, "同义改写"),
    ("手机和电脑哪个好", "手机对比电脑", 0, 0.30, "比较非等价"),
    ("退款流程", "充值流程", 0, 0.35, "字面重叠语义无关"),
    ("包邮吗", "不包邮吗", 0, 0.50, "字面极近语义对立"),
    ("能开发票", "不能开发票", 0, 0.50, "字面极近语义对立"),
]

# ======================================================================
# 二、度量实现
# ======================================================================
def ngrams(s, n=2):
    s = "".join(c for c in s if c.strip())
    return set(s[i:i+n] for i in range(max(0, len(s) - n + 1)))

def jaccard(a, b, n=2):
    A, B = ngrams(a, n), ngrams(b, n)
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def levenshtein(a, b):
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + (0 if a[i-1] == b[j-1] else 1))
            prev = tmp
    return dp[n]

def lev_sim(a, b):
    m = max(len(a), len(b), 1)
    return 1.0 - levenshtein(a, b) / m

def tokenize(s):
    # 字符 + 简单词混合：中文按字，英文/数字合并
    toks = []
    buf = ""
    for c in s:
        if c.isalnum():
            buf += c.lower()
        else:
            if buf: toks.append(buf); buf = ""
    if buf: toks.append(buf)
    # 叠加单字（中文单字也是有效 token）
    for c in s:
        if '\u4e00' <= c <= '\u9fff':
            toks.append(c)
    return toks

def tf(tokens):
    d = {}
    for t in tokens: d[t] = d.get(t, 0) + 1
    return d

def char_ngrams(s, n=3):
    s = "".join(c for c in s if c.strip())
    return [s[i:i+n] for i in range(max(0, len(s) - n + 1))]

# 全局 IDF（基于数据集构建）
IDF = {}
def build_idf(pairs):
    docs = [char_ngrams(a) + char_ngrams(b) for a, b, *_ in pairs]
    N = len(docs)
    df = {}
    for d in docs:
        for t in set(d): df[t] = df.get(t, 0) + 1
    for t, c in df.items():
        IDF[t] = math.log((N + 1) / (c + 1)) + 1

def tfidf_vec(s, n=3):
    toks = char_ngrams(s, n)
    tfd = tf(toks)
    vec = {t: c * IDF.get(t, 1.0) for t, c in tfd.items()}
    return vec

def cosine(a, b):
    if not a or not b: return 0.0
    dot = sum(a.get(t, 0) * b.get(t, 0) for t in set(a) & set(b))
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0: return 0.0
    return dot / (na * nb)

# ---- M4: PPMI 分布词向量（自构建，无需预训练模型）----
CORPUS_TOKENS = []
def build_wordvec(pairs, win=2):
    # 用所有问句构建共现
    vocab = set()
    cooc = {}
    for a, b, *_ in pairs:
        for sent in (a, b):
            toks = tokenize(sent)
            CORPUS_TOKENS.append(toks)
            vocab.update(toks)
            for i, w in enumerate(toks):
                for j in range(max(0, i - win), min(len(toks), i + win + 1)):
                    if i == j: continue
                    pair = (w, toks[j])
                    cooc[pair] = cooc.get(pair, 0) + 1
    # 频率
    freq = {}
    for toks in CORPUS_TOKENS:
        for w in toks: freq[w] = freq.get(w, 0) + 1
    total = sum(freq.values())
    # PPMI
    vecs = {w: {} for w in vocab}
    for (w, c), n in cooc.items():
        pwc = n / total
        pw = freq[w] / total
        pc = freq[c] / total
        pmi = math.log(pwc / (pw * pc)) if pwc > 0 else 0
        ppmi = max(pmi, 0)
        if ppmi > 0:
            vecs[w][c] = ppmi
    return vecs

WORDVEC = {}
def sent_vec(s, wordvec):
    toks = tokenize(s)
    dim = set()
    for t in toks:
        dim.update(wordvec.get(t, {}).keys())
    dim = list(dim)
    if not dim: return {}
    v = {}
    for t in toks:
        wv = wordvec.get(t, {})
        for d, val in wv.items():
            v[d] = v.get(d, 0) + val
    n = len(toks) or 1
    return {d: val / n for d, val in v.items()}

def wv_cos(a, b, wordvec):
    return cosine(sent_vec(a, wordvec), sent_vec(b, wordvec))

# ---- M5: 融合度量 ----
def fused(a, b, w):
    # w = (w_jac, w_lev, w_tfidf, w_wv)
    s = sum(w)
    if s == 0: return 0.0
    val = (w[0] * jaccard(a, b) + w[1] * lev_sim(a, b) + w[2] * cosine(tfidf_vec(a), tfidf_vec(b)) + w[3] * wv_cos(a, b, WORDVEC))
    return val / s

# ======================================================================
# 三、评估指标
# ======================================================================
def eval_threshold(scores, labels, th):
    tp = fp = tn = fn = 0
    for s, y in zip(scores, labels):
        pred = 1 if s >= th else 0
        if pred == 1 and y == 1: tp += 1
        elif pred == 1 and y == 0: fp += 1
        elif pred == 0 and y == 0: tn += 1
        else: fn += 1
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    acc = (tp + tn) / len(labels)
    return acc, p, r, f1

def best_f1(scores, labels):
    # 固定阈值网格搜索，排除退化（全正/全负）阈值
    best = (0, 0, 0, 0, 0)
    for th in [round(0.05 * k, 2) for k in range(1, 20)]:  # 0.05 ~ 0.95
        acc, p, r, f1 = eval_threshold(scores, labels, th)
        # 要求预测正例数在合理区间，避免退化
        pred_pos = sum(1 for s in scores if s >= th)
        if pred_pos == 0 or pred_pos == len(scores): continue
        if f1 > best[3]:
            best = (th, acc, p, r, f1)
    # 兜底：若无合法阈值，取0.5
    if best[3] == 0:
        acc, p, r, f1 = eval_threshold(scores, labels, 0.5)
        best = (0.5, acc, p, r, f1)
    return best

def auc(scores, labels):
    # 用 Mann-Whitney U 统计量近似 AUC
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg: return 0.5
    c = 0
    for ps in pos:
        for ns in neg:
            if ps > ns: c += 1
            elif ps == ns: c += 0.5
    return c / (len(pos) * len(neg))

def ap_score(scores, labels):
    # Average Precision
    order = sorted(zip(scores, labels), key=lambda x: -x[0])
    hit = 0; s = 0; total = sum(labels)
    for i, (sc, y) in enumerate(order, 1):
        if y == 1:
            hit += 1
            s += hit / i
    return s / total if total else 0

# ======================================================================
# 四、运行实验
# ======================================================================
def main():
    global WORDVEC
    t0 = time.time()
    build_idf(DATASET)
    WORDVEC = build_wordvec(DATASET, win=2)
    pos = sum(1 for p in DATASET if p[2] == 1)
    print(f"数据集规模: {len(DATASET)} 对问句 | 正例 {pos} | 负例 {len(DATASET) - pos}")
    print(f"词表规模: {len(WORDVEC)} | 耗时 {time.time()-t0:.2f}s\n")

    pairs = DATASET
    labels = [p[2] for p in pairs]
    metrics = {
        "M1 Jaccard(bigram)": lambda a, b: jaccard(a, b, 2),
        "M2 编辑距离相似度": lambda a, b: lev_sim(a, b),
        "M3 TF-IDF余弦": lambda a, b: cosine(tfidf_vec(a), tfidf_vec(b)),
        "M4 PPMI词向量余弦": lambda a, b: wv_cos(a, b, WORDVEC),
    }
    rows = []
    all_scores = {}
    for name, fn in metrics.items():
        scores = [fn(a, b) for a, b, *_ in pairs]
        all_scores[name] = scores
        b = best_f1(scores, labels)
        rows.append((name, b[0], b[1], b[2], b[3], b[4], auc(scores, labels), ap_score(scores, labels)))

    # M5 融合：在 70% 训练集搜权重，30% 测试集评估
    n = len(pairs)
    idx = list(range(n))
    split = int(n * 0.7)
    tr, te = idx[:split], idx[split:]
    # 网格搜索权重（归一化）
    best_w = (0.25, 0.25, 0.25, 0.25); best_tr_f1 = -1
    grid = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for wj, wl, wt, ww in itertools.product(grid, grid, grid, grid):
        if wj + wl + wt + ww == 0: continue
        sc = [fused(pairs[i][0], pairs[i][1], (wj, wl, wt, ww)) for i in tr]
        f1 = best_f1(sc, [labels[i] for i in tr])[3]
        if f1 > best_tr_f1:
            best_tr_f1 = f1; best_w = (wj, wl, wt, ww)
    te_scores = [fused(pairs[i][0], pairs[i][1], best_w) for i in te]
    te_labels = [labels[i] for i in te]
    b5 = best_f1(te_scores, te_labels)
    rows.append((f"M5 融合度量(权重{best_w})", b5[0], b5[1], b5[2], b5[3], b5[4], auc(te_scores, te_labels), ap_score(te_scores, te_labels)))

    # 输出表格
    header = ["度量", "最优阈值", "Acc", "P", "R", "F1", "AUC", "AP"]
    print("\t".join(header))
    for r in rows:
        print(f"{r[0]}\t{r[1]:.3f}\t{r[2]:.3f}\t{r[3]:.3f}\t{r[4]:.3f}\t{r[5]:.3f}\t{r[6]:.3f}\t{r[7]:.3f}")

    # 消融（在测试集上）
    print("\n--- 消融实验（测试集 F1/AUC，等权固定方案）---")
    ablations = {
        "仅字面(J+Lev)": (1, 1, 0, 0),
        "仅语义(TFIDF+WV)": (0, 0, 1, 1),
        "字面+TFIDF(无WV)": (1, 1, 1, 0),
        "字面+WV(无TFIDF)": (1, 1, 0, 1),
        "全融合(网格搜权重)": best_w,
    }
    for name, w in ablations.items():
        sc = [fused(pairs[i][0], pairs[i][1], w) for i in te]
        b = best_f1(sc, te_labels)
        print(f"{name}\t权重{w}\tF1={b[3]:.3f}\tAUC={auc(sc, te_labels):.3f}")

    # 阈值敏感性（融合度量）
    print("\n--- 阈值敏感性（融合度量 全数据集 F1）---")
    full_scores = [fused(pairs[i][0], pairs[i][1], best_w) for i in idx]
    for th in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        acc, p, r, f1 = eval_threshold(full_scores, labels, th)
        print(f"th={th:.2f}\tAcc={acc:.3f}\tP={p:.3f}\tR={r:.3f}\tF1={f1:.3f}")

    # 保存 JSON 结果供论文引用
    result = {
        "dataset_size": n, "positive": sum(labels), "negative": n - sum(labels),
        "weights": best_w, "train_f1": round(best_tr_f1, 3),
        "table": [dict(zip(header, [r[0], round(r[1],3), round(r[2],3), round(r[3],3), round(r[4],3), round(r[5],3), round(r[6],3), round(r[7],3)])) for r in rows],
    }
    os.makedirs("data", exist_ok=True)
    with open("data/experiment_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    # 保存数据集
    with open("data/dataset.json", "w", encoding="utf-8") as f:
        json.dump([{"a": a, "b": b, "label": l, "human_sim": hs, "case": c} for a, b, l, hs, c in DATASET], f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: data/experiment_result.json, data/dataset.json")

if __name__ == "__main__":
    main()
