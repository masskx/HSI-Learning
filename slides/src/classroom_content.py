"""Teaching revision: canonical demos, worked examples and in-class checks.

Numeric results come from artifacts/teaching/classroom-facts.json. Values from
other environments are not silently substituted or used as score targets.
"""
from pathlib import Path
import json

def enrich(lessons,page):
    root=Path(__file__).resolve().parents[2]
    facts_path=root/'artifacts/teaching/classroom-facts.json'
    if not facts_path.exists():
        raise RuntimeError('Run export_classroom_facts.py before building classroom slides')
    facts=json.loads(facts_path.read_text(encoding='utf-8'))
    results=facts.get('results',{})
    by_id={l['id']:l for l in lessons}
    def text(title,points,notes='',run=None,diagram=None,table=None):
        value=page(title,points,[],notes,run=run)
        value.update(diagram=diagram,table=table)
        return value
    def add(id,*pages): by_id[id]['pages'].extend(pages)
    # Existing figures remain traceable; legacy positional references now have IDs.
    for lesson in lessons:
        for p in lesson['pages']:
            if p['image'] and p['image'][0]=='03_hybridsn_baseline.ipynb':
                nb,cell,ordinal=p['image']; p['image']=(nb,cell.replace('cell','legacy-'),ordinal)
                p['notes']+='\n历史协议B输出，仅用于结构/读图；不是协议E的比较成绩。'
        lesson['exit_question']={
            'L00':'换kernel后，首先核对哪个路径？',
            'L01':'gt=0的像元能否存在非零光谱？',
            'L02':'OA高、AA低时，应检查混淆矩阵哪里？',
            'L03':'标准化后还应使用原始DN的gamma量级吗？',
            'L04':'shuffle后如何保证预测回到原坐标？',
            'L05':'Conv2d首层是否聚合所有输入通道？',
            'L06':'reshape合并C和D是否减少元素数？',
            'L07':'OA下降而AA上升是否表示实验失败？',
            'L08':'单episode的25个support是否等于总标签预算？',
            'L09':'未知召回100%是否足以说明系统可用？',
            'L10':'观察与候选解释各应写在哪里？'}[lesson['id']]
    by_id['L00']['pages'][0]['points'][2]='环境路径以当前kernel实际输出为准'
    by_id['L00']['pages'][4]['run']=('model(torch.zeros(2,12,9,9)).shape','torch.Size([2,16])')
    by_id['L00']['pages'][4]['notes']='先用batch2手算，再改batch3；断言与输入共享BATCH_SIZE。'
    by_id['L00']['pages'][5]['run']=None
    add('L00',text('课前资产与现场备用',['运行准备脚本，再验证四个模型包','现场不启动完整训练','网络断开时可读已执行Notebook输出'],
        '启动清单见classroom-guide。离线输出是执行记录，明确区分现场重新运行。'))
    add('L01',text('光谱曲线里的类内与类间变化',['同类不同像元也有差异','不同类别的曲线可以重叠','横轴先使用波段索引'],
        '打开Notebook01的spectral-comparison；请学生先判断哪些曲线更难区分，再展示类别。',
        diagram={'kind':'flow','labels':['同一坐标','200维观测','类别由GT监督']}),
        text('显示、标准化、PCA分别改变什么',['显示拉伸帮助人眼读图','标准化改变特征尺度','PCA改变特征表示，不新增监督标签'],
        '一张图的颜色改变不能当作模型输入已经改变；区分显示代码与训练预处理。',
        table=[['操作','拟合信息','课堂检查'],['显示拉伸','显示范围','GT和样本数不变'],['StandardScaler','训练特征','test仅transform'],['PCA/白化','训练特征','检查方差与分量数']]))
    by_id['L01']['pages'][-2]['image']=('01_data_reading_and_visualization.ipynb','spectral-comparison',0)
    by_id['L01']['pages'][-2]['diagram']=None
    by_id['L02'].update(notebook='02_svm_baseline.ipynb',cells=['protocol','metrics-toy','raw-confusion'])
    by_id['L02']['pages'][0]['run']=("len(split['test_ids'])  # 协议E",'8200')
    by_id['L02']['pages'][1]=text('信息流的方向',['训练拟合预处理与参数','验证选择超参数或checkpoint','测试只用于冻结后的报告'],
        'PCA/scaler不要先fit整图。开放集的模型选择与校准目前共享val，明确披露。',
        diagram={'kind':'flow','labels':['Train: fit','Val: select','Test: report']})
    by_id['L02']['pages'][3]['points']=['p_o与p_e都为0.9','按真实与预测边缘频率计算p_e','该构造例Kappa为0']
    by_id['L02']['pages'][2]['run']=("toy['oa'], toy['aa']",'(0.9, 0.3333)')
    by_id['L02']['pages'][3]['run']=("toy['kappa']",'0.0')
    by_id['L02']['pages'][4]['image']=('02_svm_baseline.ipynb','raw-confusion',1)
    by_id['L02']['pages'][4]['notes']='讲课图按GT类别ID标轴，名字/support表在同一Notebook cell输出；第一张完整名称图供课后阅读。颜色是行归一化recall，不能说对角线是原始数量。'
    by_id['L02']['pages'][4]['points']=['颜色=每行比例，不是原始数量','名称与样本数见课后表','小类是否全错看对应行']
    by_id['L02']['pages'][5]=text('课堂协议E与历史记录',['课堂统一seed42、10/10/80中心划分','PCA/scaler只拟合训练中心','历史A/B/C/D只作为独立记录阅读'],
        '共享split仍不等于共享输入：SVM为光谱，CNN还看空间邻域。',
        table=[['项目','课堂E'],['训练/验证/测试','1024 / 1025 / 8200'],['数据/seed','Indian Pines / 42'],['结果来源','当前run_config与metrics']])
    add('L02',text('指标分母必须写出来',['逐类recall=该类正确数/该类测试数','AA=有测试支持类别的recall平均','全图展示包含训练中心，不是test OA'],
        '没有测试样本的类标为NA并报告覆盖，不把0/0静默算成0。'))
    by_id['L03'].update(cells=['pipeline','kernel-scale','raw-confusion','pca-confusion'])
    svm=results.get('svm',{}); raw=svm.get('raw',{}).get('metrics',{})
    by_id['L03']['pages'][0]=text('先建立可重建的基线',['先检查标签是否混入特征','记录划分、输入、选参规则','分数由当前测试输出读取'],
        '协议E的SVM是本次重新执行，不能复用旧协议A的0.85作为同协议对照。',
        run=("raw_report['oa'], raw_report['aa']",f"({raw.get('oa',0):.4f}, {raw.get('aa',0):.4f})"))
    by_id['L03']['pages'][3]=text('gamma取决于模型看到的特征',['float64 exp(-500)约7.12e-218，仍非零','StandardScaler后重新理解距离尺度','原始DN网格不是标准化后的建议范围'],
        '核相似度极小与数值下溢分别解释。验证差不是核退化唯一原因。',
        run=('np.exp(-500.0) > 0','True'))
    by_id['L03']['pages'][1]['run']=("X[y > 0].shape, y[y > 0].shape",'((10249, 200), (10249,))')
    by_id['L03']['pages'][2]['notes']='算例两个相似度都非零；极小核值和数值下溢要区分。'
    by_id['L03']['pages'][5]['image']=('02_svm_baseline.ipynb','pca-confusion',1)
    by_id['L03']['pages'][5]['points'][0]='相同候选规则，先用原始200波段'
    by_id['L03']['code']="raw = joblib.load(ASSETS/'raw.joblib')\nassert (\n    raw.named_steps['standardscaler'].n_samples_seen_\n    == len(split['train_ids'])\n)\nraw_pred = raw.predict(X[split['test_ids']])"
    by_id['L03']['change']='在原始核算例中把特征乘10，再将gamma除100；另解释StandardScaler重新fit会怎样。'
    add('L03',text('验证搜索与最终报告',['每个候选仅fit训练集','按validation OA选C；并列保留较早候选','选择冻结后输出测试报告'],
        'Notebook中的RUN_FIT明确开关；课前脚本用C=[1,10,100]，完整搜索记录写入metrics。',
        diagram={'kind':'flow','labels':['Pipeline.fit(train)','比较val候选','冻结后predict(test)']}))
    by_id['L04'].update(notebook='11_patch_hybridsn_classroom.ipynb',cells=['toy-patches','reshape-check'])
    by_id['L04']['pages'][0]['image']=('11_patch_hybridsn_classroom.ipynb','toy-patches',0)
    by_id['L04']['pages'][0]['points']=['编号toy数组可手算','中心标签随原坐标保持','四角缺失观测使用明确padding']
    by_id['L04']['pages'][0]['notes']='图来自当前toy-patches：中心及四角的padding数值逐格核对。'
    by_id['L04']['pages'][1]['run']=('patches[2,0,1,1].item()','12.0')
    by_id['L04']['pages'][2]['run']=('np.zeros((9,9,12)).transpose(2,0,1).shape','(12, 9, 9)')
    by_id['L04']['code']="assert patches.shape == (5,1,3,3)\nnp.testing.assert_array_equal(\n    patches[:,0,1,1].numpy(), toy.ravel()[ids])\n# Preserve center ids when predictions are reordered."
    by_id['L04']['pages'][4]['image']=('02_svm_baseline.ipynb','maps',0)
    by_id['L04']['pages'][4]['notes']='当前协议E的SVM全图显示；test错误图另看maps的第二张输出。'
    by_id['L04']['pages'][5]['visual']=['Pair overlap 72/81','Sampling positions','Observed dependence']
    by_id['L04']['pages'][5]['points']=['横移1像元的9×9窗口共享72/81','这个数只描述一对窗口','全数据重叠需结合采样位置统计']
    by_id['L04']['pages'][5]['notes']='不能把89%说成总体泄漏量或性能增益。'
    add('L04',text('带缓冲区的空间划分',['边界两侧各留patch半径','直接检查输入窗口是否共享像元','先报告每个区域缺席的类别'],
        'Notebook02 spatial-split给出位置清单。空间划分同时改变覆盖和难度，不称净泄漏效应。',
        diagram={'kind':'flow','labels':['Train region','Radius margins','Val / test region']}))
    by_id['L05'].update(notebook='12_convolution_classroom.ipynb',cells=['manual-convolution','three-axes','receptive-field','current-curves'])
    by_id['L05']['pages'][2]['points'][2]='含池化的理论RF为26；逐层独立递推'
    by_id['L05']['pages'][3]['image']=('12_convolution_classroom.ipynb','current-curves',0)
    by_id['L05']['pages'][3]['points']=['当前协议E的train与val曲线','看趋势与验证选择，不只看末轮','逐类表现另看Notebook15']
    by_id['L05']['pages'][3]['notes']='曲线来自当前classifier-ce演示包，不沿用历史1D模型的训练数字。'
    by_id['L05']['pages'][4]['image']=('15_imbalance_analysis_teaching.ipynb','maps',0)
    by_id['L05']['pages'][4]['points']=['同split的GT与两种损失预测','全图显示包含训练中心','错误图只评价test中心']
    by_id['L05']['pages'][4]['notes']='Notebook15当前CE和weighted全图；数字只在test中心报告。'
    by_id['L05']['pages'][5]['points']=['PC是原波段的线性组合','分量相邻不等于波长相邻','RF不能换算成nm']
    by_id['L05']['pages'][5]['run']=("(cfg['components'], 200)",'(12, 200)')
    add('L05',text('一次卷积乘加',['输入[1,2,3,4,5]，核[1,0,-1]','步长1、无padding，输出长度3','三个位置输出都为-2'],
        '先写答案，再运行manual-convolution。深度学习框架的Conv实际使用互相关约定。',
        diagram={'kind':'flow','labels':['[1,2,3]','1×1+2×0−3×1','输出−2']}),
        text('三种输入轴与参数量',['batch变化不改变权重形状','2D核聚合全部Cin','3D核沿depth局部滑动'],
        '原始band与PCA分量的depth语义分别说明。',
        table=[['模型','输入','滑动轴'],['Conv1d','N,1,B','B'],['Conv2d','N,B,H,W','H,W'],['Conv3d','N,1,D,H,W','D,H,W']]))
    by_id['L06'].update(notebook='11_patch_hybridsn_classroom.ipynb',cells=['hybridsn-shapes','reshape-check','pca-change'])
    by_id['L06']['pages'][1]['run']=("shapes['conv3']",'(2,32,3,19,19)')
    by_id['L06']['pages'][2]['run']=("merged.shape",'torch.Size([2, 96, 19, 19])')
    by_id['L06']['pages'][3]['run']=("net.dense1[0].weight.shape",'torch.Size([256, 18496])')
    by_id['L06']['pages'][4]['points']=['本图保留历史协议B训练记录','权重按validation选择','当前协议E结果与历史B分开报告']
    by_id['L06']['code']="assert shapes['conv3'] == (2,32,3,19,19)\nmerged = tensor.reshape(2,96,19,19)\nassert merged.numel() == tensor.numel()"
    add('L06',text('逐层shape与元素对应',['先追踪depth和空间各自收缩','合并C和D保留元素及空间位置','25改为其他patch时重新核算FC'],
        '本课使用随机权重前向，不能把其输出命名为训练预测结果。',
        table=[['阶段','shape（N=2）'],['输入','2,1,15,25,25'],['Conv3输出','2,32,3,19,19'],['C×D合并','2,96,19,19'],['Conv2d输出','2,64,17,17']]))
    ce=results.get('classifier-ce',{}).get('metrics',{}); weighted=results.get('classifier-weighted',{}).get('metrics',{})
    by_id['L07']['pages'][1]['points']=[f"当前CE/weighted OA：{ce.get('oa',0):.4f}/{weighted.get('oa',0):.4f}",
        f"当前AA：{ce.get('aa',0):.4f}/{weighted.get('aa',0):.4f}",'加权依据频率，Focal依据预测难度']
    by_id['L07']['pages'][1]['run']=("results['CE'][1].mean()",f"{ce.get('aa',0):.4f}")
    by_id['L07']['pages'][1]['notes']='OA和AA取当前classroom-facts；固定划分的一次运行不证明普遍收益。'
    by_id['L07']['code']="response, logits = grad_cam(\n    ce, ce.encoder.features[7], inputs, torch.tensor([first]))\nassert response.shape[-2:] == inputs.shape[-2:]\nassert torch.isfinite(response).all()"
    add('L07',text('手算一个batch的加权损失',['普通CE平均每个样本的损失','weighted CE按目标权重归一化','少数类权重高，不保证所有指标改善'],
        'Notebook15 weighted-loss-toy展示与PyTorch输出核对；目标权重归一化不是简单乘后平均。',
        table=[['样本','loss','目标权重'],['大类','0.2','1'],['小类','1.0','4'],['普通/加权','0.6 / 0.84','(0.2+4)/5']]))
    by_id['L08']['pages'][0]['points']=['25仅为5-way 5-shot support标签数','训练池与训练query监督也消耗标注','验证标签与部署support另外记录']
    by_id['L08']['pages'][0]['run']=("cfg['classes']['test']",'[12, 13, 14, 15, 16]')
    by_id['L08']['cells'].append('episode-update')
    by_id['L08']['code']="logits = prototype_logits(\n    se, torch.as_tensor(ep.support_y), qe, 5)\nloss = torch.nn.functional.cross_entropy(\n    logits, torch.as_tensor(ep.query_y))\nassert not set(support) & set(query)"
    by_id['L07']['cells'].extend(['weighted-loss-toy','target-comparison'])
    add('L08',text('一个episode如何更新encoder',['support编码并建立类均值','query编码，负平方距离作为logits','query标签计算CE；反传到encoder'],
        'Notebook16 episode-update复制encoder做一次训练episode，不改变演示权重；BN统计固定。',
        diagram={'kind':'flow','labels':['Support→prototype','Query→logits→CE','backward→encoder']}),
        text('配对K对照与任务方差',['同一encoder与类集合','同一query，support按嵌套子集改变K','报告每任务差值，保留下降任务'],
        '任务间std不是新数据集的泛化置信区间。'))
    op=results.get('openset',{}); om=op.get('metrics',{}); oc=op.get('config',{})
    for index,key in [(1,'msp'),(2,'distance')]:
        tau=oc.get('thresholds',{}).get(key)
        by_id['L09']['pages'][index]['run']=(f"thresholds['{key}']",f'{tau:.4f}' if tau is not None else 'prepare assets first')
        by_id['L09']['pages'][index]['notes']='阈值来自当前run_config，已知validation校准；更换接受率必须重算指标与地图。'
    by_id['L09']['pages'][4]['points']=[f"当前AUROC MSP：{om.get('msp',{}).get('auroc',0):.3f}",
        f"当前AUROC distance：{om.get('distance',{}).get('auroc',0):.3f}",'比较固定阈值的误拒与漏检代价']
    by_id['L09']['pages'][4]['notes']='数值来自当前openset metrics；AUROC与阈值代价分别报告。'
    by_id['L09']['cells'].append('threshold-exercise')
    by_id['L09']['code']="new_tau, new_metrics, outputs = threshold_experiment(\n    true, predicted, test_dict, classes, val_dict, .90)\n# Update a COPY of cfg before drawing new maps."
    add('L09',text('接受、正确接受与误拒的分母',['已知误拒：拒掉的已知/所有已知test','已知正确接受：接受且分对/所有已知test','未知召回：拒掉的未知/所有未知test'],
        '正确接受不是接受后条件准确率。known_correct+known_wrong+known_false_rejection=1。',
        table=[['已知test去向','含义'],['拒绝','known false rejection'],['接受且正确','known correct acceptance'],['接受但错类','known wrong acceptance']]),
        text('验证集参数练习',['先选目标接受率，再查看test指标','重新校准后重算预测与错误图','AUROC不随同一分数的阈值改变'],
        'threshold-exercise不再断言新阈值等于旧阈值；原演示包不修改。'))
    by_id['L10']['pages'][0]=text('研究案例：加权损失的代价',['问题：少数类recall低，改loss是否有帮助？','唯一改动：CE变为weighted CE','比较OA、AA、逐类recall与失败地图'],
        '采用Notebook15同split证据，全部数字来自当前配置。结构和预算固定仍需更多种子，不宣称普遍优势。')
    add('L10',text('证据表到结果段落',['观察：引用配置与指标行','解释：候选机制，另提检验','限制：同场景、单seed与未对齐因素'],
        '案例和配对seed模板见research-case.md。段落不新增未经执行的数值或显著性。',
        table=[['句子角色','本课案例'],['观察','加权后OA与AA变化见Notebook15'],['候选解释','类别权重改变梯度贡献'],['限制','同场景、一次seed；不是方法上限']]))
    training=dict(id='L05A',slug='training',title='模型怎样完成一次参数更新？',subtitle='标签、logits、loss与梯度',
        prerequisites='L04；会读张量shape',notebook='10_training_step_classroom.ipynb',
        cells=['batch','forward-loss','one-update','mode-gradient','checkpoint'],assignment='A03',
        outputs=['一个batch的数据流','参数更新证据','验证checkpoint规则'],sources=['pytorch','sklearn'],
        pages=[text('从中心标签到模型target',['原始类别1…16','损失target为0…15','背景0不进入监督样本'],
                    diagram={'kind':'flow','labels':['Patch + center GT','class ID→target','logits: N×16']}),
               text('Logits与交叉熵',['argmax得到类别索引','CE内部使用log-softmax','输入loss不预先重复softmax'],
                    '用logits[2,1]手算分类0的CE约0.3133。'),
               text('一次更新的四个操作',['zero_grad清掉旧梯度','前向计算当前batch的loss','backward计算当前loss的梯度','step按优化规则更新参数'],
                    diagram={'kind':'flow','labels':['zero_grad','forward → loss','loss.backward','optimizer.step']}),
               text('模式与梯度是两套开关',['train/eval影响BN与Dropout','eval仍能计算梯度','no_grad用于无需求导的推理'],
                    table=[['操作','主要作用'],['model.eval()','固定BN/关闭Dropout随机性'],['torch.no_grad()','不记录autograd图'],['Grad-CAM','eval且允许梯度']]),
               text('验证选模型，测试做报告',['记录每轮训练与验证表现','best_epoch按validation冻结','一次batch loss下降不证明泛化提升'])],
        code="learner = deepcopy(model)\noptimizer = torch.optim.SGD(learner.parameters(), lr=.001)\noptimizer.zero_grad()\nloss = F.cross_entropy(learner(x), y)\nloss.backward()\noptimizer.step()",change='漏掉zero_grad，连续两次backward后检查梯度；只操作复制模型。',
        ai='根据给定batch代码标注输入、参数、梯度和优化器状态；为每个shape写出来源。',verify='检查目标范围、梯度存在、原演示权重未改变。',
        limits=['单batch更新不保证test改善','训练模式与梯度模式分别控制'],exit_question='eval模式是否一定没有梯度？')
    lessons.insert(next(i for i,l in enumerate(lessons) if l['id']=='L05'),training)
