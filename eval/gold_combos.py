"""B2 金标题库：60 条「人给定场景 + 概念链」的组合表。

为什么人工只给到概念层，条号由规则锁定：
* 概念链是**语义骨架**，必须人来定，否则 LLM 会造出「看似多跳」的伪题；
* 具体条号由「该概念在对应法律中 DEFINES 置信度最高且正文命中概念名」的规则选出，
  再由 ``check_gold`` 逐条回原文核验并输出抽检表 —— 规则可复现，且不冒充「人肉逐条标注」；
* 每条题带 ``intent``（一句话说明用户到底在问什么），供 LLM 只写题面。

hop_type 定义：
  two_hop   需要串联 2 个概念（每题期望命中 2 条法条）
  three_hop 需要串联 3 个概念
  cross_law 概念链跨越 ≥2 部法律
"""

from __future__ import annotations

# (id, hop_type, scenario_id, concept_chain, intent)
COMBOS: list[tuple[str, str, str, list[str], str]] = [
    # ---------------- two_hop × 30 ----------------
    ("g01", "two_hop", "unpaid_wages", ["wage_payment", "wage_deduction"],
     "公司拖欠两个月工资，员工想知道公司这样做是否违法、能不能要回工资"),
    ("g02", "two_hop", "overtime_pay_dispute", ["work_hour_extension", "overtime_pay"],
     "公司要求每天加班三小时却只按正常工资发，员工想知道加班费怎么算"),
    ("g03", "two_hop", "probation_dismissal", ["probation_period", "trial_dismissal"],
     "入职两个月还在试用期，公司说不符合录用条件就辞退，员工想知道是否合法"),
    ("g04", "two_hop", "contract_renewal", ["fixed_term_contract", "open_ended_contract"],
     "第二次固定期限合同到期，公司不愿续签，员工想知道能不能要求签无固定期限"),
    ("g05", "two_hop", "non_compete_dispute", ["non_compete", "non_compete_compensation"],
     "离职时签了竞业限制协议但公司不给补偿，员工想知道协议还有没有约束力"),
    ("g06", "two_hop", "resignation_notice", ["unilateral_rescission_by_worker", "notice_period"],
     "员工想辞职但公司要求必须干满两个月，想知道提前多久通知才合法"),
    ("g07", "two_hop", "immediate_resignation", ["immediate_rescission", "social_insurance"],
     "公司一直不缴社保，员工想马上走人并要求补偿"),
    ("g08", "two_hop", "economic_layoff", ["economic_layoff", "severance_standard"],
     "公司一次性裁掉三十人，员工想知道裁员程序和经济补偿怎么算"),
    ("g09", "two_hop", "illegal_dismissal", ["illegal_rescission", "damages"],
     "员工没犯错却被突然开除，想知道能拿到多少赔偿"),
    ("g10", "two_hop", "double_wage_claim", ["written_form", "double_wage"],
     "上班半年公司一直没签书面合同，员工想知道能不能主张双倍工资"),
    ("g11", "two_hop", "annual_leave", ["annual_leave", "rest_period"],
     "工作三年从未休年假，员工想知道年休假天数和未休补偿"),
    ("g12", "two_hop", "working_hours_dispute", ["working_hours", "rest_day"],
     "公司实行单休且每天工作十小时，员工想知道工时和休息日是否合规"),
    ("g13", "two_hop", "wage_deduction", ["wage_deduction", "labor_quota"],
     "公司按所谓内部定额扣钱，员工想知道这种扣工资是否合法"),
    ("g14", "two_hop", "minimum_wage", ["minimum_wage", "minimum_wage_standard"],
     "员工月工资低于当地最低工资标准，想知道差额能否补"),
    ("g15", "two_hop", "part_time_job", ["part_time_employment", "part_time_wage_settlement"],
     "员工每天做四小时的小时工，公司月结一次工资，想知道是否合法"),
    ("g16", "two_hop", "labor_dispatch", ["labor_dispatch", "accepting_unit"],
     "员工被派遣到另一家公司上班，出了纠纷不知道该找谁"),
    ("g17", "two_hop", "contractor_vs_employee", ["labor_relationship", "written_form"],
     "员工没签书面合同但正常出勤，想知道是否构成劳动关系、公司不签书面合同要承担什么责任"),
    ("g18", "two_hop", "collective_contract", ["collective_contract", "labor_union"],
     "公司要改薪酬方案，员工想知道工会和集体合同能不能约束公司"),
    ("g19", "two_hop", "confidentiality_breach", ["confidentiality", "trade_secret"],
     "员工离职后把客户名单带走，公司想知道能否追责"),
    ("g20", "two_hop", "service_period_training", ["service_period", "training"],
     "公司出钱送员工培训并约定服务期，员工提前离职要不要赔"),
    ("g21", "two_hop", "sick_leave", ["sick_leave", "contract_termination"],
     "员工长期病假，公司停发工资并想解除合同，想知道医疗期和解除限制"),
    ("g22", "two_hop", "employment_discrimination", ["employment_discrimination", "equal_employment_right"],
     "招聘启事写明不招女性，求职者想知道这算不算就业歧视"),
    ("g23", "two_hop", "minor_worker", ["minor_worker", "child_labor"],
     "店里想招十五岁的孩子做帮工，店主想知道是否违法"),
    ("g24", "two_hop", "handover_dispute", ["handover", "separation_certificate"],
     "员工离职时公司扣押档案、不给离职证明，想知道该怎么办"),
    ("g25", "two_hop", "deposit_dispute", ["deposit", "breach_of_contract"],
     "买方付了定金后反悔，想知道定金能不能退"),
    ("g26", "two_hop", "lease_dispute", ["lease_contract", "deposit"],
     "租客退租时房东以各种理由不退押金"),
    ("g27", "two_hop", "loan_dispute", ["loan_contract", "breach_of_contract"],
     "朋友借钱到期不还，出借人想知道能否主张逾期利息"),
    ("g28", "two_hop", "sales_breach", ["sales_contract", "damages_civil"],
     "买家收到的货物质量不合格，想知道能要求退货还是赔偿"),
    ("g29", "two_hop", "guarantee_dispute", ["guarantee", "guarantee_contract"],
     "朋友借款让员工做保证人，出借人想知道保证责任范围"),
    ("g30", "two_hop", "tort_medical", ["medical_malpractice_liability", "fault_liability"],
     "患者手术后出现并发症，家属想知道医院是否要担责"),

    # ---------------- three_hop × 20 ----------------
    ("g31", "three_hop", "unpaid_wages", ["wage", "wage_payment", "wage_deduction"],
     "公司连续欠薪，员工想知道工资支付义务、拖欠的认定以及可以向谁投诉"),
    ("g32", "three_hop", "overtime_pay_dispute", ["working_hours", "work_hour_extension", "overtime_pay"],
     "公司长期安排超时加班又不付加班费，员工想知道工时上限、加班认定和加班费标准"),
    ("g33", "three_hop", "probation_dismissal", ["probation_period", "trial_dismissal", "illegal_rescission"],
     "试用期最后一天被以不符合录用条件辞退，员工想知道试用期限制、辞退条件以及是否属于违法解除"),
    ("g34", "three_hop", "economic_layoff", ["economic_layoff", "labor_union", "severance"],
     "公司未通知工会就宣布裁员，员工想知道裁员程序、工会作用和补偿"),
    ("g35", "three_hop", "non_compete_dispute", ["non_compete", "non_compete_compensation", "liquidated_damages"],
     "员工签了竞业协议，离职后公司不付补偿还要求他赔违约金，员工想知道协议还有没有约束力"),
    ("g36", "three_hop", "social_insurance_dispute", ["labor_relationship", "social_insurance", "pension_insurance"],
     "公司按最低基数缴社保，员工担心退休后养老金受影响"),
    ("g37", "three_hop", "maternity_leave", ["female_worker_protection", "labor_protection", "illegal_rescission"],
     "女员工怀孕后被调岗降薪并辞退，想知道三期劳动保护以及这是否属于违法解除"),
    ("g38", "three_hop", "labor_dispatch", ["labor_dispatch", "dispatch_unit", "accepting_unit"],
     "派遣工在用工单位受伤，派遣公司和用工单位都不愿担责"),
    ("g39", "three_hop", "part_time_job", ["part_time_employment", "part_time_wage_settlement", "labor_dispute"],
     "小时工被拖欠工资，想知道结算周期规定以及怎么维权"),
    ("g40", "three_hop", "contract_invalid", ["invalid_contract", "fraud", "coercion"],
     "员工入职时被要求签空白合同并被威胁，想知道合同是否无效"),
    ("g41", "three_hop", "service_period_training", ["training", "service_period", "liquidated_damages"],
     "培训协议约定的违约金高于培训费，员工想知道违约金上限"),
    ("g42", "three_hop", "sick_leave", ["sick_leave", "contract_termination", "severance"],
     "员工医疗期满后公司终止合同，想知道是否要给补偿"),
    ("g43", "three_hop", "arbitration_procedure", ["labor_dispute", "arbitration", "limitation_period"],
     "员工离职一年半后才想申请仲裁，担心是否已过时效"),
    ("g44", "three_hop", "guarantee_dispute", ["guarantee_contract", "joint_liability", "security_interest"],
     "保证人被要求承担全部债务，想知道保证方式和担保物受偿顺序"),
    ("g45", "three_hop", "lease_dispute", ["lease_contract", "deposit", "damages_civil"],
     "租客提前退租，房东扣下押金并主张损失赔偿"),
    ("g46", "three_hop", "tort_traffic", ["traffic_accident_liability", "no_fault_liability", "damages_civil"],
     "交通事故中无责方受伤，想知道赔偿范围和无过错情形下的责任"),
    ("g47", "three_hop", "tort_product", ["product_liability", "no_fault_liability", "sales_contract"],
     "消费者买到缺陷家电受伤，想知道找商家还是厂家、能否要求惩罚性赔偿"),
    ("g48", "three_hop", "privacy_infringe", ["right_of_privacy", "personal_information", "network_tort"],
     "公司把员工人脸信息卖给第三方并在网上公开，员工想维权"),
    ("g49", "three_hop", "inheritance_dispute", ["inheritance", "statutory_inheritance", "estate_debt"],
     "老人去世后既有遗嘱又有债务，继承人想知道如何分割和清偿"),
    ("g50", "three_hop", "agency_dispute", ["agency", "apparent_agency", "unauthorized_agency"],
     "业务员越权签了合同，公司想不认账，对方主张表见代理"),

    # ---------------- cross_law × 10 ----------------
    ("g51", "cross_law", "contract_terms_missing", ["labor_contract", "contract_establishment", "standard_terms"],
     "劳动合同里的竞业条款是公司单方拟定的格式条款，员工想知道效力和合同成立规则"),
    ("g52", "cross_law", "severance_calculation", ["liquidated_damages", "liquidated_damages_civil", "breach_of_contract"],
     "劳动合同约定员工违约要赔十万，员工想知道违约金能否调整"),
    ("g53", "cross_law", "illegal_dismissal", ["illegal_rescission", "damages", "damages_civil"],
     "被违法辞退后员工同时主张劳动赔偿金和一般民事损害赔偿"),
    ("g54", "cross_law", "arbitration_procedure", ["labor_dispute", "arbitration", "limitation_period"],
     "劳动仲裁与普通民事纠纷的时效规则有何不同"),
    ("g55", "cross_law", "work_injury", ["work_injury", "tort_liability", "no_fault_liability"],
     "员工工伤后向第三人主张侵权赔偿，想知道能否双重获赔"),
    ("g56", "cross_law", "part_time_job", ["part_time_employment", "offer", "contract_establishment"],
     "小时工没有书面合同，双方对报酬口头约定，想知道合同是否成立"),
    ("g57", "cross_law", "contract_invalid", ["invalid_contract", "fraud", "coercion"],
     "公司用虚假材料骗员工签合同，签字时还施加威胁，员工想知道合同是不是无效"),
    ("g58", "cross_law", "occupational_disease", ["labor_protection", "tort_liability", "fault_liability"],
     "员工因公司未提供防护得职业病，想知道既走工伤又能否主张侵权"),
    ("g59", "cross_law", "economic_layoff", ["severance", "breach_of_contract", "damages_civil"],
     "公司裁员只给一个月补偿，还要求员工支付违约金、赔偿损失，员工想知道补偿和赔偿分别怎么算"),
    ("g60", "cross_law", "platform_worker", ["labor_relationship", "agency", "apparent_agency"],
     "平台骑手由外包公司管理，出事后平台是否承担用工责任"),
]

# B7 第 2 步：人工逐条核对 60 条金标后，对「概念→条文」错映射的修正。
# 只改映射，不改场景/概念链/题面；判定依据是「该概念在对应法律中的规制条款 + 题面诉求」，
# **不看检索结果**（避免按结果反向造数据）。每条修正的 before→after 与理由落盘
# data/eval_out/gold_article_review.json。
#
# 约束：修正后的条文必须与概念在图里有真实边（否则路径走不通）。若正确条文因字面匹配漏抽，
# 在 data/lexicon/manual_mentions.jsonl 里按法律含义补一条 provenance=manual 的边并写理由，
# 不放宽匹配规则去凑数（例：劳动法第三十八条写「每周至少休息一日」，没有「休息日」三字）。
ARTICLE_OVERRIDES: dict[str, dict[str, str]] = {
    # g06 主动辞职的预告期：劳动者单方解除的核心条文是第三十七条，原映射第六十五条是派遣解除
    "g06": {"unilateral_rescission_by_worker": "lcl#37"},
    # g07 社保欠缴：救济条款是「劳动者可解除」（第三十八条），劳动法第 70 条只是社保事业总述
    "g07": {"social_insurance": "lcl#38"},
    # g10 未签书面合同：责任基础是第十条（应当订立书面合同、一个月内），第四十条是无过失辞退
    "g10": {"written_form": "lcl#10"},
    # g11 年休假：天数与未休补偿落在第四十五条；第三条只是权利清单
    "g11": {"rest_period": "ll#45"},
    # g12 单休是否合法：劳动法第三十八条（每周至少休息一日），原映射第四十四条是加班费标准
    "g12": {"rest_day": "ll#38"},
    # g16 派遣纠纷找谁：第五十八条（派遣单位是用人单位）+ 第九十二条（连带赔偿责任）
    "g16": {"labor_dispatch": "lcl#58", "accepting_unit": "lcl#92"},
    # g19 保密/商业秘密追责：第二十三条（保密约定）+ 第九十条（违反保密义务赔偿）
    "g19": {"confidentiality": "lcl#90", "trade_secret": "lcl#23"},
    # g25 定金反悔：第五百八十七条（定金罚则）+ 第五百七十七条（违约责任总则）
    "g25": {"deposit": "cc#587", "breach_of_contract": "cc#577"},
    # g26 退租押金：第七百三十三条（返还租赁物）+ 第五百八十七条（定金罚则）
    "g26": {"lease_contract": "cc#733", "deposit": "cc#587"},
    # g27 借款不还：第六百八十条（利息约定）+ 第五百七十七条（违约责任）
    "g27": {"loan_contract": "cc#680", "breach_of_contract": "cc#577"},
    # g28 质量不合格能否退货：第六百一十条（不符合质量要求致合同目的不能实现，可拒收或解除）
    #     原映射第六百三十五条是凭样品买卖，与普通货物质量纠纷无关
    #     damages_civil 取第五百八十四条（违约损失赔偿额的计算），精排会漂到第一千一百八十三条（精神损害）
    "g28": {"sales_contract": "cc#610", "damages_civil": "cc#584"},
    # g30 手术并发症：过错责任总则第一千一百六十五条，而非委托合同过错
    "g30": {"fault_liability": "cc#1165"},
    # g31 欠薪：第三十条（及时足额支付、支付令），而非工资分配原则
    "g31": {"wage": "lcl#30"},
    # g34 裁员未通知工会：第四十三条（单方解除应事先通知工会）
    "g34": {"labor_union": "lcl#43"},
    # g35 竞业违约金：第二十三条（竞业限制违约金），而非民法典违约金调整条款
    "g35": {"liquidated_damages": "lcl#23"},
    # g36 社保缴费：第七十二条（必须依法参保缴费）+ 第七十三条（退休后享受社保待遇）
    #     + 劳动合同法第七条（用工之日起建立劳动关系）；原映射第十六条只讲「应当订立合同」
    "g36": {"social_insurance": "ll#72", "pension_insurance": "ll#73", "labor_relationship": "lcl#7"},
    # g37 孕期降薪辞退：劳动法第五十八条明确「国家对女职工实行特殊劳动保护」，
    #     原映射第五十二条是通用安全卫生制度，与三期保护无关
    "g37": {"labor_protection": "ll#58"},
    # g38 派遣工伤找谁：第五十八条（派遣单位义务）+ 第九十二条（连带赔偿）
    "g38": {"labor_dispatch": "lcl#58", "dispatch_unit": "lcl#58", "accepting_unit": "lcl#92"},
    # g39 小时工工资结算：第七十二条（结算周期），而非口头协议条款
    "g39": {"part_time_employment": "lcl#72"},
    # g40 欺诈胁迫致合同无效：劳动合同法第二十六条直接规定无效，民法典第 148/150 条是可撤销
    "g40": {"fraud": "lcl#26", "coercion": "lcl#26"},
    # g42 医疗期满终止与补偿：第四十四条（终止情形）+ 第四十六条（经济补偿情形）
    "g42": {"contract_termination": "lcl#44", "severance": "lcl#46"},
    # g43 仲裁时效：劳动法第八十二条（六十日申请期限）+ 民法典第一百九十八条（仲裁时效从 labor 法规定）
    "g43": {"labor_dispute": "ll#82", "arbitration": "ll#82", "limitation_period": "cc#198"},
    # g44 物保与人保并存：第三百九十二条（有约定按约定，无约定先就物的担保受偿），
    #     原映射第三百八十八条只是担保合同总则，回答不了受偿顺序
    "g44": {"security_interest": "cc#392"},
    # g45 提前退租扣留押金：第七百三十三条（交还租赁物）+ 第五百八十七条（定金抵作价款或收回），
    #     原映射第七百二十五条是「买卖不破租赁」，与押金/损失无关
    "g45": {"lease_contract": "cc#733", "deposit": "cc#587"},
    # g50 越权代理：第一百七十一条（无权代理不生效力）+ 第一百七十二条（表见代理）
    "g50": {"agency": "cc#171"},
    # g51 格式条款效力：劳动合同法第二十六条（免除自己责任、排除劳动者权利的条款无效）
    #     + 劳动合同法第十六条（签字或盖章生效，即劳动合同场景下的成立规则）；原映射第十三条只是固定期限定义
    "g51": {"labor_contract": "lcl#26", "contract_establishment": "lcl#16"},
    # g55 工伤待遇：第七十三条（因工伤残享受社保待遇），第二十九条是「不得解除」
    "g55": {"work_injury": "ll#73"},
    # g58 职业病侵权：过错责任总则第一千一百六十五条
    "g58": {"fault_liability": "cc#1165"},
    # g59 裁员补偿与违约金：第四十七条（补偿标准）+ 第五百八十五条（违约金与损失赔偿额的计算方法）
    "g59": {"severance": "lcl#47", "breach_of_contract": "cc#585", "damages_civil": "cc#585"},
    # g60 平台表见代理：第一百七十二条（表见代理），第一百六十五条是授权委托书形式要件
    #     + 劳动合同法第七条（用工之日起建立劳动关系）；原映射第十六条只讲「应当订立合同」
    "g60": {"agency": "cc#172", "labor_relationship": "lcl#7"},
    # g56 小时工口头协议：第四百七十六条（要约可否撤销）+ 第四百八十三条（承诺生效时合同成立）；
    #     原映射第四百七十二条是定义、第四百九十条要求「合同书」，都不回答口头协议
    "g56": {"offer": "cc#476", "contract_establishment": "cc#483"},
}

# 未修正、仍留作「如实未通过」的条目及原因（写进 gold_article_review.json）
UNRESOLVED: dict[str, str] = {
    "g26": "「押金」在链里被建模为「定金」，但与退租押金的法律性质不同；租赁侧无「提前解除/返还押金」的字面边",
    "g46": "「无过错责任」只连到第一千一百六十六条，无偿搭乘减责条文第一千二百一十七条无字面边",
    "g55": "「工伤」在劳动法里只有第七十三条（享受社保待遇）与第二十九条（不得解除）两条边，都不直接回答「工伤与侵权能否双重获赔」",
}

# 每条修正的理由（键为 (gold_id, concept_id)），与上面 ARTICLE_OVERRIDES 一一对应。
OVERRIDE_REASONS: dict[tuple[str, str], str] = {
    ("g06", "unilateral_rescission_by_worker"): "劳动者提前三十日书面通知即可解除，第三十七条；原映射第六十五条只适用于被派遣劳动者",
    ("g07", "social_insurance"): "社保欠缴的救济是「劳动者可解除」，第三十八条直接规定；劳动法第七十条只是社保事业总述",
    ("g10", "written_form"): "未签书面合同的责任基础是第十条（应当订立书面合同、一个月内），第四十条是无过失辞退",
    ("g11", "rest_period"): "年休假天数与未休补偿落在第四十五条；第三条只是权利清单",
    ("g12", "rest_day"): "每周至少休息一日是休息日的正面规制，劳动法第三十八条；原映射第四十四条是加班费标准",
    ("g16", "labor_dispatch"): "派遣纠纷先要确定派遣单位是用人单位，第五十八条",
    ("g16", "accepting_unit"): "用工单位与派遣单位承担连带赔偿责任，第九十二条",
    ("g19", "confidentiality"): "违反保密义务造成损失应赔偿，第九十条；第二十三条只讲「可以约定」",
    ("g19", "trade_secret"): "商业秘密的保密约定与竞业限制都在第二十三条",
    ("g25", "deposit"): "给付定金方不履行无权请求返还定金，第五百八十七条",
    ("g25", "breach_of_contract"): "违约责任总则，第五百七十七条",
    ("g26", "lease_contract"): "租赁期限届满承租人应返还租赁物，第七百三十三条",
    ("g26", "deposit"): "定金罚则，第五百八十七条",
    ("g27", "loan_contract"): "借款利息约定不明的处理，第六百八十条",
    ("g27", "breach_of_contract"): "违约责任总则，第五百七十七条",
    ("g28", "sales_contract"): "质量不符合要求致合同目的不能实现时可拒收或解除，第六百一十条；原映射是凭样品买卖",
    ("g28", "damages_civil"): "违约造成损失的赔偿额按第五百八十四条计算；精排按字面选了第一千一百八十三条（精神损害），与「退货 + 赔偿损失」的诉求不符",
    ("g36", "pension_insurance"): "退休后依法享受社会保险待遇，劳动法第七十三条第一项；原映射第四十四条讲的是劳动合同终止",
    ("g36", "labor_relationship"): "用人单位自用工之日起即与劳动者建立劳动关系，劳动合同法第七条",
    ("g30", "fault_liability"): "过错责任总则，第一千一百六十五条；原映射是委托合同过错，与医疗场景无关",
    ("g31", "wage"): "及时足额支付劳动报酬与支付令，第三十条；原映射是工资分配原则",
    ("g34", "labor_union"): "单方解除应事先将理由通知工会，第四十三条",
    ("g35", "liquidated_damages"): "竞业限制违约金由劳动合同法第二十三条规定",
    ("g36", "social_insurance"): "用人单位和劳动者必须依法参加社会保险、缴纳社保费，劳动法第七十二条",
    ("g38", "labor_dispatch"): "劳务派遣单位是用人单位，应履行用人单位义务，第五十八条",
    ("g38", "dispatch_unit"): "同上，第五十八条",
    ("g38", "accepting_unit"): "给被派遣劳动者造成损害的，派遣单位与用工单位承担连带赔偿责任，第九十二条",
    ("g39", "part_time_employment"): "非全日制用工的工资结算周期，第七十二条",
    ("g40", "fraud"): "以欺诈手段订立的劳动合同无效，劳动合同法第二十六条直接规定，优先于民法典可撤销条款",
    ("g40", "coercion"): "以胁迫手段订立的劳动合同无效，第二十六条",
    ("g42", "contract_termination"): "劳动合同终止的法定情形，第四十四条",
    ("g42", "severance"): "用人单位应支付经济补偿的情形，第四十六条",
    ("g43", "labor_dispute"): "劳动争议仲裁申请期限，劳动法第八十二条",
    ("g43", "arbitration"): "同上，第八十二条",
    ("g37", "labor_protection"): "国家对女职工实行特殊劳动保护，劳动法第五十八条；原映射第五十二条是通用安全卫生制度",
    ("g43", "limitation_period"): "法律对仲裁时效有规定的依照其规定，民法典第一百九十八条；劳动争议仲裁适用劳动法第八十二条的六十日",
    ("g44", "security_interest"): "同一债权既有物的担保又有人的担保时如何受偿，第三百九十二条；原映射第三百八十八条只是担保合同总则",
    ("g45", "lease_contract"): "租赁关系结束时交还租赁物，第七百三十三条；原映射第七百二十五条是买卖不破租赁",
    ("g45", "deposit"): "定金应当抵作价款或者收回，第五百八十七条；原映射第五百八十六条只讲定金合同成立与上限",
    ("g51", "labor_contract"): "免除用人单位法定责任、排除劳动者权利的条款无效，劳动合同法第二十六条",
    ("g51", "contract_establishment"): "劳动合同经双方签字或盖章生效，劳动合同法第十六条；劳动合同场景优先适用特别法，而非民法典第四百八十三条",
    ("g56", "offer"): "要约可以撤销及其例外，第四百七十六条，对应「对方能不能反悔」",
    ("g56", "contract_establishment"): "承诺生效时合同成立，第四百八十三条，对应口头协议是否成立；原映射第四百九十条要求合同书",
    ("g50", "agency"): "无权代理未经追认不发生效力，第一百七十一条",
    ("g55", "work_injury"): "因工伤残依法享受社会保险待遇，劳动法第七十三条；原映射是「不得解除」",
    ("g58", "fault_liability"): "过错责任总则，第一千一百六十五条",
    ("g59", "severance"): "经济补偿按工作年限计算，第四十七条",
    ("g59", "breach_of_contract"): "违约金数额及过高/过低时的调整，第五百八十五条，对应「违约金怎么算」",
    ("g59", "damages_civil"): "违约金与损失赔偿额的计算方法同条规定于第五百八十五条，与「违约金」并条，避免两条重复",
    ("g60", "agency"): "表见代理的效力，第一百七十二条；原映射是授权委托书形式要件",
    ("g60", "labor_relationship"): "用人单位自用工之日起即与劳动者建立劳动关系，劳动合同法第七条",
}
