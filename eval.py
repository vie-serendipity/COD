# -*- coding: utf-8 -*-
# @Time    : 2020/11/21
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os

import cv2
from tqdm import tqdm

# pip install pysodmetrics
from metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

FM = Fmeasure()
WFM = WeightedFmeasure()
SM = Smeasure()
EM = Emeasure()
MAE = MAE()

data_root = "/Data/ZZY/TestDataset/COD10K/GT"
pred_root = "/Data/ZZY/P_Edge_N/snapshot/exp9/COD10K"
mask_root = os.path.join(data_root)
pred_root = os.path.join(pred_root)
mask_name_list = sorted(os.listdir(mask_root))
for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
    mask_path = os.path.join(mask_root, mask_name)
    pred_path = os.path.join(pred_root, mask_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    FM.step(pred=pred, gt=mask)
    WFM.step(pred=pred, gt=mask)
    SM.step(pred=pred, gt=mask)
    EM.step(pred=pred, gt=mask)
    MAE.step(pred=pred, gt=mask)

fm = FM.get_results()["fm"]
wfm = WFM.get_results()["wfm"]
sm = SM.get_results()["sm"]
em = EM.get_results()["em"]
mae = MAE.get_results()["mae"]

results = {
    "Smeasure": sm,
    "wFmeasure": wfm,
    "MAE": mae,
    "adpEm": em["adp"],
    "meanEm": em["curve"].mean(),
    "maxEm": em["curve"].max(),
    "adpFm": fm["adp"],
    "meanFm": fm["curve"].mean(),
    "maxFm": fm["curve"].max(),
}

print(results)
# 'Smeasure': 0.9029763868504661,
# 'wFmeasure': 0.5579812753638986,
# 'MAE': 0.03705558476661653,
# 'adpEm': 0.9408760066970631,
# 'meanEm': 0.9566258293508715,
# 'maxEm': 0.966954482892271,
# 'adpFm': 0.5816750824038355,
# 'meanFm': 0.577051059518767,
# 'maxFm': 0.5886784581120638

# version 1.2.3
# 'Smeasure': 0.9029763868504661,
# 'wFmeasure': 0.5579812753638986,
# 'MAE': 0.03705558476661653,
# 'adpEm': 0.9408760066970631,
# 'meanEm': 0.9566258293508715,
# 'maxEm': 0.966954482892271,
# 'adpFm': 0.5816750824038355,
# 'meanFm': 0.577051059518767,
# 'maxFm': 0.5886784581120638

  # 0 带测试集
# 1 CAMO训练集加COD10K训练集 使用边缘引导，四个特征图求和
COD10K={'Smeasure': 0.779827592357242, 'wFmeasure': 0.6393669713702586, 
'MAE': 0.03603390408466672, 'adpEm': 0.8385548485776061,
 'meanEm': 0.8172988941717537, 'maxEm': 0.8353059204251357, 
 'adpFm': 0.6890091581949287, 'meanFm': 0.6797183519931865, 'maxFm': 0.6854943202842261}
CAMO={'Smeasure': 0.7780837873788936, 'wFmeasure': 0.6893123988550239, 
'MAE': 0.07051740909272289, 'adpEm': 0.8349461331166553, 
'meanEm': 0.8267254398772044, 'maxEm': 0.8344047347352896, 
'adpFm': 0.7400823875704635, 'meanFm': 0.7304248302051752, 'maxFm': 0.7355777646341595}
CHAMELEON = {'Smeasure': 0.8311116273566606, 'wFmeasure': 0.7505532198713603,
 'MAE': 0.03408091773421489, 'adpEm': 0.8824988639873727, 
 'meanEm': 0.8691728308618935, 'maxEm': 0.883050192687039,
  'adpFm': 0.7924581560471232, 'meanFm': 0.7852510662565667, 'maxFm': 0.7915905999032748}
# 2 不使用边缘引导，四个特征图求和
COD10K={'Smeasure': 0.7692332031841558, 'wFmeasure': 0.6205252889332046,
 'MAE': 0.03691837744018854, 'adpEm': 0.8259710212013641, 
 'meanEm': 0.8041425852142462, 'maxEm': 0.8243843020734712,
  'adpFm': 0.6719232205316125, 'meanFm': 0.6626530102138759, 'maxFm': 0.668393971584632}
CAMO = {'Smeasure': 0.7629744580023078, 'wFmeasure': 0.6656728569038814, 
'MAE': 0.07866014070455031, 'adpEm': 0.8157954715936846, 
'meanEm': 0.8010192961147771, 'maxEm': 0.8256086674298654, 
'adpFm': 0.7273585243674586, 'meanFm': 0.7104252215406781, 'maxFm': 0.7157298357357526}
CHAMELEON = {'Smeasure': 0.8358421962523248, 'wFmeasure': 0.7606366827283184, 
'MAE': 0.035516579667952984, 'adpEm': 0.9029197379884458,
 'meanEm': 0.8873831820707176, 'maxEm': 0.9028069785962469, 
 'adpFm': 0.7997532716105, 'meanFm': 0.7956770522903157, 'maxFm': 0.8032318670149401}
# 3 边缘引导 使用ds_map3
COD10K={'Smeasure': 0.7795235473687339, 'wFmeasure': 0.6339131745719966, 
'MAE': 0.03715653897576691, 'adpEm': 0.8504070002690198, 
'meanEm': 0.8244799213964122, 'maxEm': 0.8449565854746973, 
'adpFm': 0.6756647349918327, 'meanFm': 0.6742597121553606, 'maxFm': 0.6820616056639418}
CAMO={'Smeasure': 0.7802821369990481, 'wFmeasure': 0.6858582399761856, 
'MAE': 0.07207666225840506, 'adpEm': 0.8402469828253667,
 'meanEm': 0.8270187078817164, 'maxEm': 0.8373205837590652, 
 'adpFm': 0.7364267011379656,'meanFm': 0.7263274021838473, 'maxFm': 0.733657490989145}
CHAMELEON = {'Smeasure': 0.8333297108873575, 'wFmeasure': 0.7466473365303208, 
 'MAE': 0.035340648370180895, 'adpEm': 0.8830122519247935, 
 'meanEm': 0.8722842976102558, 'maxEm': 0.8856434393431529, 
 'adpFm': 0.77975340809815, 'meanFm': 0.7796001591190139, 'maxFm': 0.7886472939612723}
 # 4 不使用边缘引导 使用ds_map3
COD10K = {'Smeasure': 0.7687812971676388, 'wFmeasure': 0.6203058778783952, 
'MAE': 0.03683124178153417, 'adpEm': 0.8232421091023567,
 'meanEm': 0.8029978339613335, 'maxEm': 0.8219707148672751, 
 'adpFm': 0.6705721536038469, 'meanFm': 0.6623873438739987, 'maxFm': 0.6681399638996451}
CAMO={'Smeasure': 0.7627991906391206, 'wFmeasure': 0.6658485123002447, 
'MAE': 0.07849282339396042, 'adpEm': 0.814618110885005, 
'meanEm': 0.8012745303400479, 'maxEm': 0.8231970972860522, 
'adpFm': 0.7271270789643005, 'meanFm': 0.7105234508685572, 'maxFm': 0.71518388018636}
CHAMELEON={'Smeasure': 0.8353832431765911, 'wFmeasure': 0.7606922692045229,
 'MAE': 0.03543918044985366, 'adpEm': 0.9023704843592233, 
 'meanEm': 0.8873021416164012, 'maxEm': 0.9019833922714253, 
 'adpFm': 0.7998523436304176, 'meanFm': 0.7959077321913068, 'maxFm': 0.803184802791806}

# 5 使用gra引导各三次
COD10K = {'Smeasure': 0.7930459661212953, 'wFmeasure': 0.663405896840139, 
'MAE': 0.03400149169727645, 'adpEm': 0.8557275668965654, 
'meanEm': 0.8409249979415389, 'maxEm': 0.8522416483463114,
 'adpFm': 0.7005911850414668, 'meanFm': 0.7023053415522964, 'maxFm': 0.7096107172859283}
CAMO = {'Smeasure': 0.7931114562195237, 'wFmeasure': 0.7224472675590353, 
'MAE': 0.06977632566719322, 'adpEm': 0.8583741292020363, 
'meanEm': 0.8519629102288997, 'maxEm': 0.8587489168516402, 
'adpFm': 0.7714435914197061, 'meanFm': 0.7627055189598366, 'maxFm': 0.7676152426221465}
# 6 使用gra各两次
# worse
# 7 使用深层的三层
COD10K = {'Smeasure': 0.7670835655716678, 'wFmeasure': 0.6202816793332178, 
'MAE': 0.0380331115860345, 'adpEm': 0.8188947626320973, 
'meanEm': 0.8056413470015168, 'maxEm': 0.8182186892514556, 
'adpFm': 0.6701480401937464, 'meanFm': 0.6599691744865012, 'maxFm': 0.6651765031090214}
# 8 五层
{'Smeasure': 0.7797900777662766, 'wFmeasure': 0.6421910334132794,
 'MAE': 0.03644489023455649, 'adpEm': 0.8280615034648374,
  'meanEm': 0.8178049372832206, 'maxEm': 0.8274578586497808, 
  'adpFm': 0.6867007866787052, 'meanFm': 0.6817256748590761, 'maxFm': 0.6878513265238336}
# 9 前三层 引导三次, rfb后，但是没有detach
{'Smeasure': 0.7695094935718172, 'wFmeasure': 0.6196239947748717, 
'MAE': 0.03706582042877573, 'adpEm': 0.8140957562433689, 
'meanEm': 0.7967773302758088, 'maxEm': 0.8118274760700555, 
'adpFm': 0.6637539217166858, 'meanFm': 0.658775060481642, 'maxFm': 0.6640953865470236}
# COD10K-CAM-1-Aquatic-18-StarFish-1161
# COD10K-CAM-3-Flying-55-Butterfly-3318
# COD10K-CAM-3-Flying-57-Dragonfly-3590