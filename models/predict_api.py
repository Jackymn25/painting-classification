import csv
import random
import re

import numpy as np
import pandas as pd

# model parameters

W = [[-0.47829150000523263, 1.0269104714468489, -0.5486189857109306], [0.025476832706715812, 0.03569997635650507, -0.06117680906321506], [0.5794121854924644, -0.05659590081374247, -0.5228162846787207], [-0.5604602803996346, 0.07119876079925971, 0.48926151960038394], [-0.005077850780599368, -0.01650242594830385, 0.021580276728906], [0.7540173121692848, -0.233263739787194, -0.5207535723820828], [0.332644583445543, -0.20569727595887483, -0.12694730748667035], [0.24054427410552964, 0.17900985077547252, -0.41955412488101024], [-0.12998695137492777, 0.20060473525302563, -0.07061778387809672], [-0.08146923141788895, -0.14946323899292735, 0.23093247041081888], [-0.5961262881722899, 0.4326896177181379, 0.16343667045414487], [-0.11739427146811621, -0.0071820320372485745, 0.1245763035053619], [-0.23683673994994672, 0.12832510194320504, 0.10851163800674285], [0.14774497772354245, -0.03265710678899395, -0.1150878709345457], [0.18982516196208946, -0.09704921893895968, -0.0927759430231297], [-0.20027653098933962, 0.05398035380134289, 0.1462961771879991], [-0.2998335100600482, 0.0977401235582664, 0.20209338650177797], [0.04722353204060018, 0.2189603499947765, -0.266183882035372], [0.022517722853217632, 0.04378525676249285, -0.06630297961571127], [0.6167995229193491, -0.030786913051220586, -0.5860126098681169], [-0.3852197801298104, -0.3578040507636376, 0.7430238308934488], [-0.04405304302400622, 0.02165173926215311, 0.022401303761849848], [-0.00806370931304881, 0.7718949154632829, -0.7638312061502324], [0.3352116969894839, 0.08053494706921129, -0.4157466440586909], [-0.0008001098930609126, 0.30569830396824815, -0.30489819407518526], [-0.25132280432559795, -0.3246118444960961, 0.5759346488216893], [-0.13934059541391042, 0.2656769282277239, -0.12633633281381873], [-0.16742480362028486, 0.07378892558139472, 0.0936358780388838], [0.030511736058225093, 0.13759480701960786, -0.16810654307783232], [0.6138257637660063, -0.3181554820450937, -0.2956702817209146], [-0.3314642355962941, 0.6512771534171082, -0.3198129178208112], [-0.42905002428584027, -0.17014552380358947, 0.5991955480894295], [0.47360765005706, -0.12011699741782149, -0.3534906526392403], [-0.07355708098107731, -0.03301778081703512, 0.10657486179811386], [-0.16157995198908578, 0.05887436400606059, 0.1027055879830249], [0.25943115090993846, 0.024676083031929457, -0.28410723394187376], [0.07914432151009111, 0.29745548934468663, -0.37659981085477945], [-0.30058649809242516, -0.2230557994062232, 0.5236422974986481], [0.23525879666053842, 0.1731088715202976, -0.4083676681808417], [0.05530579410120908, 0.14088945020640337, -0.19619524430761323], [-0.07276879745397884, 0.21441312301553453, -0.14164432556155074]]

MU = [6.3354037267080745, 2.775776397515528, 3.384472049689441, 3.672049689440994, 2.441614906832298, 3.801863354037267, 3.946583850931677, 4794972.155354037, 0.253416149068323, 0.4118012422360248, 0.29751552795031055, 0.5391304347826087, 0.391304347826087, 0.5956521739130435, 0.24285714285714285, 0.5273291925465838, 0.5658385093167702, 0.22981366459627328, 0.2732919254658385, 0.3795031055900621, 0.40869565217391307, 0.2515527950310559, 0.3082285693458137, 0.3527197593411341, 0.33905167131305214, 0.6881987577639752, 0.9962732919254659, 0.32919254658385094, 0.34284738196917636, 0.3140001016582959, 0.34315251637252786, 0.8683229813664596, 3.2801242236024843, 0.1422360248447205, 0.33747951645776475, 0.3274119256443944, 0.33510855789784094, 0.8161490683229814, 2.9745341614906833, 0.1360248447204969]

SIGMA = [2.184769696289433, 1.376388125850879, 1.2846469900211237, 1.2201319132494313, 1.3878379899995033, 2.3025498166915987, 3.195824742275723, 34913372.78097509, 0.43496713032102147, 0.4921595057792663, 0.4571652202199534, 0.49846645732057443, 0.48804226784007926, 0.4907653834835021, 0.42880945770867523, 0.49925255656306033, 0.49564633630336213, 0.42071289992238753, 0.4456494686870086, 0.4852633289643472, 0.4915928356557416, 0.43390550393273364, 0.21995902646688506, 0.2527287127196692, 0.2925744362280343, 0.7957479299054239, 0.9900052198091744, 0.46992000793485056, 0.35947247226741114, 0.25790864931457863, 0.31503545948669875, 0.873375856299146, 2.810214727892983, 0.34929205270245206, 0.26975435885351046, 0.18965535895582955, 0.2827566408505722, 0.8066417325002057, 2.6157615781883026, 0.3428149447431684]

FEATURE_COLUMNS = ['On a scale of 1–10, how intense is the emotion conveyed by the artwork?', 'This art piece makes me feel sombre.', 'This art piece makes me feel content.', 'This art piece makes me feel calm.', 'This art piece makes me feel uneasy.', 'How many prominent colours do you notice in this painting?', 'How many objects caught your eye in the painting?', 'How much (in Canadian dollars) would you be willing to pay for this painting?', 'if_you_could_purchase_th__bathroom', 'if_you_could_purchase_th__bedroom', 'if_you_could_purchase_th__dining_room', 'if_you_could_purchase_th__living_room', 'if_you_could_purchase_th__office', 'if_you_could_view_this_a__by_yourself', 'if_you_could_view_this_a__coworkers_classmates', 'if_you_could_view_this_a__family_members', 'if_you_could_view_this_a__friends', 'if_you_could_view_this_a__strangers', 'what_season_does_this_ar__fall', 'what_season_does_this_ar__spring', 'what_season_does_this_ar__summer', 'what_season_does_this_ar__winter', 'food_nb_p0', 'food_nb_p1', 'food_nb_p2', 'food_nb_pred', 'food_nb_hit_count', 'food_nb_zero_hit', 'feeling_nb_p0', 'feeling_nb_p1', 'feeling_nb_p2', 'feeling_nb_pred', 'feeling_nb_hit_count', 'feeling_nb_zero_hit', 'soundtrack_nb_p0', 'soundtrack_nb_p1', 'soundtrack_nb_p2', 'soundtrack_nb_pred', 'soundtrack_nb_hit_count', 'soundtrack_nb_zero_hit']

FOOD_VOCAB = ['salad', 'ice', 'cream', 'blueberry', 'soup', 'cake', 'bread', 'pizza', 'like', 'cheese', 'chocolate', 'pie', 'sandwich', 'cold', 'noodle', 'fruit', 'thi', 'green', 'spaghetti', 'painting', 'strawberry', 'apple', 'fresh', 'pasta', 'bowl', 'food', 'warm', 'cheesecake', 'chicken', 'tea', 'steak', 'hot', 'matcha', 'rice', 'sweet', 'light', 'blue', 'egg']

FOOD_CLASS_COUNT = [541.0, 535.0, 534.0]

FOOD_FEATURE_PROB = [[0.004612546125461255, 0.04151291512915129, 0.03782287822878229, 0.006457564575645757, 0.0507380073800738, 0.017527675276752766, 0.08025830258302583, 0.06549815498154982, 0.035977859778597784, 0.07472324723247233, 0.026752767527675275, 0.023062730627306273, 0.026752767527675275, 0.03228782287822878, 0.024907749077490774, 0.004612546125461255, 0.017527675276752766, 0.004612546125461255, 0.021217712177121772, 0.015682656826568265, 0.0009225092250922509, 0.01014760147601476, 0.0027675276752767526, 0.015682656826568265, 0.011992619926199263, 0.023062730627306273, 0.01014760147601476, 0.008302583025830259, 0.01014760147601476, 0.0027675276752767526, 0.006457564575645757, 0.01014760147601476, 0.0009225092250922509, 0.024907749077490774, 0.004612546125461255, 0.0009225092250922509, 0.004612546125461255, 0.013837638376383764], [0.012126865671641791, 0.12406716417910447, 0.10914179104477612, 0.1501865671641791, 0.07742537313432836, 0.05503731343283582, 0.013992537313432836, 0.021455223880597014, 0.036380597014925374, 0.013992537313432836, 0.05690298507462686, 0.0457089552238806, 0.013992537313432836, 0.03451492537313433, 0.03264925373134328, 0.0009328358208955224, 0.01958955223880597, 0.0065298507462686565, 0.03264925373134328, 0.021455223880597014, 0.0046641791044776115, 0.013992537313432836, 0.008395522388059701, 0.03451492537313433, 0.027052238805970148, 0.017723880597014924, 0.028917910447761194, 0.03264925373134328, 0.01958955223880597, 0.0046641791044776115, 0.030783582089552237, 0.025186567164179104, 0.0009328358208955224, 0.008395522388059701, 0.010261194029850746, 0.010261194029850746, 0.027052238805970148, 0.0046641791044776115], [0.2850467289719626, 0.04018691588785047, 0.04579439252336449, 0.002803738317757009, 0.02149532710280374, 0.05700934579439252, 0.01588785046728972, 0.02149532710280374, 0.03457943925233645, 0.01588785046728972, 0.002803738317757009, 0.01588785046728972, 0.03457943925233645, 0.002803738317757009, 0.0065420560747663555, 0.05700934579439252, 0.025233644859813085, 0.04953271028037383, 0.0065420560747663555, 0.02149532710280374, 0.05327102803738318, 0.03271028037383177, 0.04579439252336449, 0.0065420560747663555, 0.01588785046728972, 0.014018691588785047, 0.014018691588785047, 0.010280373831775701, 0.017757009345794394, 0.04018691588785047, 0.0065420560747663555, 0.0065420560747663555, 0.038317757009345796, 0.0065420560747663555, 0.02149532710280374, 0.02336448598130841, 0.0009345794392523365, 0.012149532710280374]]

FEELING_VOCAB = ['feel', 'mak', 'painting', 'calm', 'thi', 'like', 'time', 'happy', 'peaceful', 'sense', 'relaxed', 'sad', 'feeling', 'clock', 'sky', 'giv', 'life', 'quiet', 'world', 'warm', 'bit', 'content', 'look', 'nostalgic', 'little', 'night', 'uneasy', 'peace', 'nature', 'remind', "i'm", "it'", 'melting', 'confused', 'away', 'awe', 'beautiful', 'bright', 'serene', 'wonder', 'way', 'hopeful', 'looking', 'color']

FEELING_CLASS_COUNT = [541.0, 535.0, 534.0]

FEELING_FEATURE_PROB = [[0.5691881918819188, 0.46586715867158673, 0.33302583025830257, 0.045202952029520294, 0.253690036900369, 0.2333948339483395, 0.4308118081180812, 0.0027675276752767526, 0.023062730627306273, 0.09132841328413284, 0.015682656826568265, 0.1540590405904059, 0.08394833948339483, 0.1522140221402214, 0.004612546125461255, 0.052583025830258305, 0.0544280442804428, 0.03044280442804428, 0.045202952029520294, 0.011992619926199263, 0.0507380073800738, 0.006457564575645757, 0.035977859778597784, 0.03782287822878229, 0.048892988929889296, 0.0027675276752767526, 0.09501845018450185, 0.01014760147601476, 0.011992619926199263, 0.03966789667896679, 0.03782287822878229, 0.045202952029520294, 0.08210332103321033, 0.06549815498154982, 0.07287822878228782, 0.0027675276752767526, 0.004612546125461255, 0.0009225092250922509, 0.008302583025830259, 0.004612546125461255, 0.026752767527675275, 0.006457564575645757, 0.017527675276752766, 0.021217712177121772], [0.45615671641791045, 0.38899253731343286, 0.21175373134328357, 0.2826492537313433, 0.17444029850746268, 0.14272388059701493, 0.036380597014925374, 0.058768656716417914, 0.09235074626865672, 0.07742537313432836, 0.04757462686567164, 0.03264925373134328, 0.043843283582089554, 0.0009328358208955224, 0.1259328358208955, 0.03451492537313433, 0.02332089552238806, 0.0457089552238806, 0.043843283582089554, 0.01585820895522388, 0.0457089552238806, 0.027052238805970148, 0.03451492537313433, 0.028917910447761194, 0.03824626865671642, 0.09794776119402986, 0.0046641791044776115, 0.036380597014925374, 0.017723880597014924, 0.025186567164179104, 0.03451492537313433, 0.01585820895522388, 0.0009328358208955224, 0.01585820895522388, 0.0046641791044776115, 0.07369402985074627, 0.03451492537313433, 0.030783582089552237, 0.021455223880597014, 0.0625, 0.025186567164179104, 0.03451492537313433, 0.028917910447761194, 0.017723880597014924], [0.3990654205607477, 0.3130841121495327, 0.19906542056074766, 0.32242990654205606, 0.16542056074766356, 0.14485981308411214, 0.014018691588785047, 0.3317757009345794, 0.15046728971962617, 0.04018691588785047, 0.13738317757009347, 0.0009345794392523365, 0.04018691588785047, 0.0009345794392523365, 0.0009345794392523365, 0.03457943925233645, 0.038317757009345796, 0.03644859813084112, 0.019626168224299065, 0.0794392523364486, 0.008411214953271028, 0.07196261682242991, 0.03457943925233645, 0.038317757009345796, 0.01588785046728972, 0.0009345794392523365, 0.0009345794392523365, 0.05327102803738318, 0.06448598130841121, 0.02336448598130841, 0.014018691588785047, 0.025233644859813085, 0.0009345794392523365, 0.0009345794392523365, 0.002803738317757009, 0.002803738317757009, 0.04018691588785047, 0.04392523364485981, 0.04579439252336449, 0.004672897196261682, 0.017757009345794394, 0.027102803738317756, 0.02149532710280374, 0.027102803738317756]]

SOUNDTRACK_VOCAB = ['slow', 'soundtrack', 'music', 'sound', 'piano', 'calm', 'like', 'melody', 'soft', 'song', 'violin', 'quiet', 'feel', 'background', 'peaceful', 'upbeat', 'classical', 'light', 'happy', 'wind', 'bird', 'string', 'instrument', 'gentle', 'sad', 'thi', 'rhythm', 'time', 'low', 'maybe', 'piece', 'calming', 'tempo', 'chirping', 'flowing', 'fast', 'flute', 'nature', 'playing', 'track', 'painting', 'soothing', 'guitar', 'noise', 'high', 'bright', 'long', 'sense', 'sombre', 'ambient', 'feeling', 'imagine', 'relaxing', 'slightly', 'warm', 'instrumental', 'key', 'ton', 'water', 'beat', 'eerie', 'tone']

SOUNDTRACK_CLASS_COUNT = [541.0, 535.0, 534.0]

SOUNDTRACK_FEATURE_PROB = [[0.36808118081180813, 0.21863468634686348, 0.13191881918819187, 0.18357933579335795, 0.10977859778597786, 0.07841328413284133, 0.1282287822878229, 0.07656826568265683, 0.048892988929889296, 0.07103321033210332, 0.08394833948339483, 0.07841328413284133, 0.07841328413284133, 0.058118081180811805, 0.013837638376383764, 0.004612546125461255, 0.03966789667896679, 0.008302583025830259, 0.0027675276752767526, 0.0470479704797048, 0.004612546125461255, 0.03228782287822878, 0.035977859778597784, 0.015682656826568265, 0.09317343173431734, 0.03228782287822878, 0.03413284132841329, 0.07472324723247233, 0.07656826568265683, 0.04151291512915129, 0.026752767527675275, 0.013837638376383764, 0.03413284132841329, 0.0027675276752767526, 0.0027675276752767526, 0.011992619926199263, 0.008302583025830259, 0.011992619926199263, 0.035977859778597784, 0.03966789667896679, 0.021217712177121772, 0.013837638376383764, 0.015682656826568265, 0.04151291512915129, 0.01937269372693727, 0.004612546125461255, 0.052583025830258305, 0.021217712177121772, 0.03966789667896679, 0.03044280442804428, 0.026752767527675275, 0.023062730627306273, 0.008302583025830259, 0.043357933579335796, 0.013837638376383764, 0.026752767527675275, 0.024907749077490774, 0.024907749077490774, 0.011992619926199263, 0.021217712177121772, 0.0470479704797048, 0.026752767527675275], [0.14085820895522388, 0.18003731343283583, 0.18003731343283583, 0.12779850746268656, 0.17257462686567165, 0.16511194029850745, 0.10354477611940298, 0.07369402985074627, 0.08488805970149253, 0.05317164179104478, 0.07555970149253731, 0.08115671641791045, 0.0457089552238806, 0.05503731343283582, 0.043843283582089554, 0.030783582089552237, 0.0625, 0.028917910447761194, 0.012126865671641791, 0.051305970149253734, 0.0065298507462686565, 0.05503731343283582, 0.036380597014925374, 0.030783582089552237, 0.02332089552238806, 0.0457089552238806, 0.030783582089552237, 0.021455223880597014, 0.01958955223880597, 0.01958955223880597, 0.049440298507462684, 0.04197761194029851, 0.027052238805970148, 0.008395522388059701, 0.028917910447761194, 0.03451492537313433, 0.010261194029850746, 0.008395522388059701, 0.027052238805970148, 0.01585820895522388, 0.03451492537313433, 0.030783582089552237, 0.01585820895522388, 0.021455223880597014, 0.025186567164179104, 0.0065298507462686565, 0.0065298507462686565, 0.021455223880597014, 0.017723880597014924, 0.01585820895522388, 0.021455223880597014, 0.021455223880597014, 0.027052238805970148, 0.012126865671641791, 0.013992537313432836, 0.017723880597014924, 0.01585820895522388, 0.013992537313432836, 0.0009328358208955224, 0.01958955223880597, 0.008395522388059701, 0.021455223880597014], [0.11121495327102804, 0.1897196261682243, 0.1691588785046729, 0.15794392523364487, 0.15794392523364487, 0.14485981308411214, 0.10934579439252337, 0.09439252336448598, 0.10934579439252337, 0.08691588785046729, 0.04579439252336449, 0.03271028037383177, 0.06261682242990654, 0.04579439252336449, 0.09252336448598131, 0.10934579439252337, 0.03271028037383177, 0.09626168224299066, 0.11495327102803739, 0.02897196261682243, 0.11495327102803739, 0.038317757009345796, 0.04953271028037383, 0.07383177570093458, 0.002803738317757009, 0.04018691588785047, 0.04766355140186916, 0.010280373831775701, 0.008411214953271028, 0.04018691588785047, 0.025233644859813085, 0.04018691588785047, 0.0308411214953271, 0.07757009345794393, 0.05700934579439252, 0.04018691588785047, 0.06822429906542056, 0.06635514018691589, 0.02336448598130841, 0.02897196261682243, 0.02336448598130841, 0.03271028037383177, 0.04392523364485981, 0.012149532710280374, 0.027102803738317756, 0.05700934579439252, 0.008411214953271028, 0.02336448598130841, 0.008411214953271028, 0.017757009345794394, 0.01588785046728972, 0.019626168224299065, 0.02897196261682243, 0.0065420560747663555, 0.03457943925233645, 0.01588785046728972, 0.019626168224299065, 0.02149532710280374, 0.04579439252336449, 0.01588785046728972, 0.0009345794392523365, 0.0065420560747663555]]

MULTIHOT_ENCODERS = {'If you could purchase this painting, which room would you put that painting in?': ['Bathroom', 'Bedroom', 'Dining room', 'Living room', 'Office'], 'If you could view this art in person, who would you want to view it with?': ['By yourself', 'Coworkers/Classmates', 'Family members', 'Friends', 'Strangers'], 'What season does this art piece remind you of?': ['Fall', 'Spring', 'Summer', 'Winter']}

# =========================
# Paths / Columns
# =========================
TRAIN_CSV_CANDIDATES = [
    "original.csv",
]

# Optional: set this to a RAW-format test CSV path when you want
# __main__ to clean it, predict it, and report accuracy.
TEST_CSV_PATH = "test.csv"

LABEL_COL = "label"
UNIQUE_ID_COL = "unique_id"

FOOD_COL = "If this painting was a food, what would be?"
FEEL_COL = "Describe how this painting makes you feel."
SOUND_COL = "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting."

ROOM_COL = "If you could purchase this painting, which room would you put that painting in?"
WHO_COL = "If you could view this art in person, who would you want to view it with?"
SEASON_COL = "What season does this art piece remind you of?"

MULTIHOT_COLS = [ROOM_COL, WHO_COL, SEASON_COL]

INTENSITY_COL = "On a scale of 1–10, how intense is the emotion conveyed by the artwork?"
PROMINENT_COLOURS_COL = "How many prominent colours do you notice in this painting?"
OBJECTS_COL = "How many objects caught your eye in the painting?"
PAYMENT_COL = "How much (in Canadian dollars) would you be willing to pay for this painting?"

LIKERT_COLS = [
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy.",
]

LIKERT_MAP = {
    "1 - Strongly disagree": 1,
    "2 - Disagree": 2,
    "3 - Neutral/Unsure": 3,
    "3 - Neutral": 3,
    "4 - Agree": 4,
    "5 - Strongly agree": 5,
}

# =========================
# Best params
# =========================
FOOD_K = 38
FEELING_K = 44
SOUNDTRACK_K = 62
NB_ALPHA = 0.5

# Original tuning log on the cleaned 1610x17 CSV:
# C=0.1 -> mean_val_acc=0.9124
# C=0.5 -> mean_val_acc=0.9124
# We use the slightly more regularized of the tied best values.
LOGREG_C = 0.5

MIN_TOKEN_LEN = 2
EXCLUDE_MISSING_FROM_VOCAB = True
ZERO_HIT_RETURNS_PRIOR = True
SORT_TIES_ALPHABETICALLY = True

# Small numerical damping for the intercept block in the Newton solve.
# This makes the full-K softmax Hessian non-singular while keeping behaviour
# essentially identical to sklearn multinomial logistic regression.
INTERCEPT_DAMPING = 1e-6
NEWTON_MAX_ITER = 25
NEWTON_TOL = 1e-8

# Validation config
RUN_KFOLD_VALIDATION_IN_MAIN = True
KFOLD_N_SPLITS = 5
KFOLD_SHUFFLE = True
KFOLD_RANDOM_STATE = 42

LABEL_TO_NAME = {
    0: "The Persistence of Memory",
    1: "The Starry Night",
    2: "The Water Lily Pond",
}

ALIASES = {
    "blueberries": "blueberry",
    "noodles": "noodle",
    "violins": "violin",
    "slowly": "slow",
    "icecream": "ice",
}

def normalize_text(text):
    if pd.isna(text):
        return ""
    s = str(text).strip().lower()
    s = s.replace("\xa0", " ")
    s = s.replace("–", "-").replace("—", "-")
    s = " ".join(s.split())
    return s

def collapse_spaced_thousands(text):
    if not text:
        return text

    pattern = re.compile(r"(?<!\d)(\d{1,3}(?: \d{3})+)(?!\d)")

    prev = None
    s = text
    while prev != s:
        prev = s
        s = pattern.sub(lambda m: m.group(1).replace(" ", ""), s)

    return s

def contains_any(text, patterns):
    for pat in patterns:
        if re.search(pat, text):
            return True
    return False

def looks_like_uncertain_text(s):
    uncertain_patterns = [
        r"\bnot sure\b",
        r"\bunsure\b",
        r"\bidk\b",
        r"\bi don't know\b",
        r"\bcannot decide\b",
        r"\bcan't decide\b",
        r"\bno idea\b",
    ]
    return contains_any(s, uncertain_patterns)

def looks_like_zero_text(s):
    zero_patterns = [
        r"\b0+\b",
        r"\bzero\b",
        r"\bnothing\b",
        r"\bno money\b",
        r"\bfree\b",
    ]
    return contains_any(s, zero_patterns)

def scale_multiplier(scale):
    scale = (scale or "").lower().strip()

    if scale in {"k", "thousand"}:
        return 1_000
    elif scale in {"mil", "million"}:
        return 1_000_000
    elif scale in {"b", "billion"}:
        return 1_000_000_000
    else:
        return 1

def score_candidate(context):
    score = 0

    realistic_patterns = [
        r"\bmax\b",
        r"\bat most\b",
        r"\bno more than\b",
        r"\bwould pay\b",
        r"\bi'd pay\b",
        r"\bwilling to pay\b",
        r"\bbecause i'm not\b",
        r"\bbecause i am not\b",
        r"\brealistically\b",
    ]

    hypothetical_patterns = [
        r"\bif i were\b",
        r"\bif i was\b",
        r"\bif i had\b",
        r"\bbillionaire\b",
        r"\bin a perfect world\b",
    ]

    if contains_any(context, realistic_patterns):
        score += 5

    if contains_any(context, hypothetical_patterns):
        score -= 5

    return score

def choose_best_payment_value(candidates, full_text):
    if not candidates:
        return np.nan

    scored = []
    for idx, item in enumerate(candidates):
        scored.append((score_candidate(item["context"]), idx, item["value"]))

    best_score = max(x[0] for x in scored)

    if best_score > 0:
        best_group = [x for x in scored if x[0] == best_score]
        best_group.sort(key=lambda x: x[1])
        return best_group[-1][2]

    if re.search(r"\bdepends\b", full_text) or re.search(r"\bif i were\b", full_text):
        return candidates[-1]["value"]

    return max(item["value"] for item in candidates)

def parse_money_value(text):
    if pd.isna(text):
        return np.nan

    s = normalize_text(text)

    if s in {"", "nan", "n/a", "na", "none"}:
        return np.nan

    s = collapse_spaced_thousands(s)

    if looks_like_zero_text(s):
        return 0.0

    num_pattern = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    scale_pattern = r"(?:k|thousand|mil|million|b|billion)?"

    money_pattern = re.compile(
        rf"""
        (?:
            (?:cad\s*\$?|\$|usd\s*\$?)\s*
            (?P<num1>{num_pattern})
            \s*(?P<scale1>{scale_pattern})
        )
        |
        (?:
            (?P<num2>{num_pattern})
            \s*(?P<scale2>{scale_pattern})
            \s*(?:cad|usd|dollars?|bucks?)\b
        )
        |
        (?:
            (?P<num3>{num_pattern})
            \s+(?P<scale3>k|thousand|mil|million|b|billion)\b
        )
        |
        (?:
            (?P<num4>{num_pattern})(?P<scale4>k|mil|b)\b
        )
        |
        (?:
            \b(?P<num5>{num_pattern})\b
        )
        """,
        re.VERBOSE | re.IGNORECASE
    )

    candidates = []

    for match in money_pattern.finditer(s):
        num = None
        scale = ""

        if match.group("num1") is not None:
            num = match.group("num1")
            scale = match.group("scale1") or ""
        elif match.group("num2") is not None:
            num = match.group("num2")
            scale = match.group("scale2") or ""
        elif match.group("num3") is not None:
            num = match.group("num3")
            scale = match.group("scale3") or ""
        elif match.group("num4") is not None:
            num = match.group("num4")
            scale = match.group("scale4") or ""
        elif match.group("num5") is not None:
            num = match.group("num5")
            scale = ""

        if num is None:
            continue

        num_clean = num.replace(",", "")

        try:
            value = float(num_clean)
        except ValueError:
            continue

        value *= scale_multiplier(scale)

        start = max(0, match.start() - 35)
        end = min(len(s), match.end() + 35)
        context = s[start:end]

        candidates.append({
            "value": value,
            "context": context
        })

    if candidates:
        return choose_best_payment_value(candidates, s)

    if looks_like_uncertain_text(s):
        return np.nan

    return np.nan

def parse_payment_column(series):
    x = series.apply(parse_money_value)

    negative_count = int((x < 0).sum())
    x.loc[x < 0] = 0

    cap_value = 324_000_000
    capped_count = int((x > cap_value).sum())
    x = x.clip(lower=0, upper=cap_value)

    return x, negative_count, cap_value, capped_count

def clean_count_column(series, upper_q=0.99, fixed_upper=None):
    # x = pd.to_numeric(series, errors="coerce")
    #
    # negative_count = int((x < 0).sum())
    # x.loc[x < 0] = 0
    #
    # non_na = x.dropna()
    # if len(non_na) == 0:
    #     return x, negative_count, None, 0
    #
    # upper = int(np.ceil(non_na.quantile(upper_q)))
    # before = x.copy()
    # x = x.clip(lower=0, upper=upper)
    # x = x.round()
    #
    # capped_count = int((before > x).sum())
    # return x, negative_count, upper, capped_count
    x = pd.to_numeric(series, errors="coerce")
    x = x.where(np.isfinite(x), np.nan)

    negative_count = int((x < 0).sum())
    x.loc[x < 0] = 0

    non_na = x.dropna()
    if len(non_na) == 0:
        return x, negative_count, fixed_upper, 0

    if fixed_upper is not None:
        upper = fixed_upper
    else:
        upper = int(np.ceil(non_na.quantile(upper_q)))

    before = x.copy()
    x = x.clip(lower=0, upper=upper)
    x = x.round()

    capped_count = int((before > x).sum())
    return x, negative_count, upper, capped_count

def clean_bounded_scale(series, lower, upper):
    x = pd.to_numeric(series, errors="coerce")
    invalid_count = int(((x < lower) | (x > upper)).sum())
    x = x.where((x >= lower) & (x <= upper), np.nan)
    return x, invalid_count

def safe_cast_to_nullable_int(series, lower=None, upper=None, default_if_all_nan=None):
    x = pd.to_numeric(series, errors="coerce")
    x = x.where(np.isfinite(x), np.nan)

    if lower is not None or upper is not None:
        x = x.clip(lower=lower, upper=upper)

    int_info = np.iinfo(np.int64)
    x = x.clip(lower=int_info.min, upper=int_info.max)

    if default_if_all_nan is not None and x.notna().sum() == 0:
        x = x.fillna(default_if_all_nan)

    x = x.round()
    return pd.array(x, dtype="Int64")

def ensure_raw_schema(df_raw):
    df = df_raw.copy()
    expected_text = [
        "Painting",
        FEEL_COL,
        ROOM_COL,
        WHO_COL,
        SEASON_COL,
        FOOD_COL,
        SOUND_COL,
    ]
    expected_numeric_like = [
        UNIQUE_ID_COL,
        INTENSITY_COL,
        PROMINENT_COLOURS_COL,
        OBJECTS_COL,
        PAYMENT_COL,
    ] + LIKERT_COLS
    for col in expected_text:
        if col not in df.columns:
            df[col] = np.nan
    for col in expected_numeric_like:
        if col not in df.columns:
            df[col] = np.nan
    return df

def clean_raw_dataframe(df_raw, drop_rows_with_many_missing=True):
    """
    Reconstruct the cleaned dataframe using the old cleaning logic.

    Training-time behavior:
      - drop rows with >= 4 missing values, matching the old pipeline.

    Prediction-time behavior:
      - keep all rows and impute/fill them using the same default rules,
        so predict/predict_all never silently delete user-provided rows.
    """
    df = ensure_raw_schema(df_raw)

    if "Painting" in df.columns and LABEL_COL not in df.columns:
        label_map = {
            "The Persistence of Memory": 0,
            "The Starry Night": 1,
            "The Water Lily Pond": 2
        }
        df["Painting"] = df["Painting"].astype(str).str.strip()
        df["Painting"] = df["Painting"].replace("nan", np.nan)
        df[LABEL_COL] = df["Painting"].map(label_map)

    for col in LIKERT_COLS:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace("nan", np.nan)
        df[col] = df[col].map(LIKERT_MAP)

    df[PAYMENT_COL], _, _, _ = parse_payment_column(df[PAYMENT_COL])

    # count_cols = [PROMINENT_COLOURS_COL, OBJECTS_COL]
    # for col in count_cols:
    #     df[col], _, _, _ = clean_count_column(df[col], upper_q=0.99)
    count_cols = [PROMINENT_COLOURS_COL, OBJECTS_COL]

    df[PROMINENT_COLOURS_COL], _, _, _ = clean_count_column(
        df[PROMINENT_COLOURS_COL], upper_q=0.99, fixed_upper=20
    )

    df[OBJECTS_COL], _, _, _ = clean_count_column(
        df[OBJECTS_COL], upper_q=0.99, fixed_upper=50
    )

    df[INTENSITY_COL], _ = clean_bounded_scale(df[INTENSITY_COL], 1, 10)

    for col in LIKERT_COLS:
        df[col], _ = clean_bounded_scale(df[col], 1, 5)

    missing_count = df.isna().sum(axis=1)
    if drop_rows_with_many_missing:
        df_clean = df[missing_count < 4].copy()
    else:
        df_clean = df.copy()

    # Payment gets its own default fill rule first, just like the old pipeline.
    payment_non_na = df_clean[PAYMENT_COL].dropna()
    if len(payment_non_na) == 0:
        payment_median = 100.0
    else:
        payment_median = payment_non_na.median()
    df_clean[PAYMENT_COL] = df_clean[PAYMENT_COL].fillna(payment_median)

    numeric_cols = [INTENSITY_COL, PROMINENT_COLOURS_COL, OBJECTS_COL] + LIKERT_COLS
    numeric_default_fills = {
        INTENSITY_COL: 5.0,
        PROMINENT_COLOURS_COL: 3.0,
        OBJECTS_COL: 3.0,
    }
    for col in LIKERT_COLS:
        numeric_default_fills[col] = 3.0

    for col in numeric_cols:
        non_na = df_clean[col].dropna()
        if len(non_na) == 0:
            global_median = numeric_default_fills[col]
        else:
            global_median = non_na.median()
        df_clean[col] = df_clean[col].fillna(global_median)

    # discrete_int_cols = [INTENSITY_COL] + count_cols + LIKERT_COLS
    # for col in discrete_int_cols:
    #     df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").round().astype("Int64")
    df_clean[INTENSITY_COL] = safe_cast_to_nullable_int(
        df_clean[INTENSITY_COL], lower=1, upper=10, default_if_all_nan=5
    )

    df_clean[PROMINENT_COLOURS_COL] = safe_cast_to_nullable_int(
        df_clean[PROMINENT_COLOURS_COL], lower=0, upper=20, default_if_all_nan=3
    )

    df_clean[OBJECTS_COL] = safe_cast_to_nullable_int(
        df_clean[OBJECTS_COL], lower=0, upper=50, default_if_all_nan=3
    )

    for col in LIKERT_COLS:
        df_clean[col] = safe_cast_to_nullable_int(
            df_clean[col], lower=1, upper=5, default_if_all_nan=3
        )

    text_or_categorical_cols = [
        "Painting",
        FEEL_COL,
        ROOM_COL,
        WHO_COL,
        SEASON_COL,
        FOOD_COL,
        SOUND_COL,
    ]
    for col in text_or_categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("Missing")

    return df_clean.reset_index(drop=True)

def resolve_training_dataframe():
    # Prefer an already-cleaned CSV because that is what the original reported
    # 0.9124 validation results were run on.
    last_exc = None

    script_dir_parts = re.split(r"[\/]", __file__)
    script_dir = "" if len(script_dir_parts) <= 1 else "/".join(script_dir_parts[:-1])

    for path in TRAIN_CSV_CANDIDATES:
        candidate_paths = [path]
        if script_dir:
            candidate_paths.append(script_dir + "/" + path)

        for candidate in candidate_paths:
            try:
                df = pd.read_csv(candidate)
                if LABEL_COL in df.columns:
                    return df, candidate, True
                return clean_raw_dataframe(df), candidate, False
            except Exception as e:
                last_exc = e
                continue

    raise FileNotFoundError("Could not load any training CSV candidate.") from last_exc

# =========================
# Text preprocessing / NB
# =========================
stop_words = {
    "'d",
    "'ll",
    "'m",
    "'re",
    "'s",
    "'ve",
    "a",
    "about",
    "above",
    "across",
    "after",
    "afterwards",
    "again",
    "against",
    "all",
    "almost",
    "alone",
    "along",
    "already",
    "also",
    "although",
    "always",
    "am",
    "among",
    "amongst",
    "amoungst",
    "amount",
    "an",
    "and",
    "another",
    "any",
    "anyhow",
    "anyone",
    "anything",
    "anyway",
    "anywhere",
    "are",
    "around",
    "as",
    "at",
    "back",
    "be",
    "became",
    "because",
    "become",
    "becomes",
    "becoming",
    "been",
    "before",
    "beforehand",
    "behind",
    "being",
    "below",
    "beside",
    "besides",
    "between",
    "beyond",
    "bill",
    "both",
    "bottom",
    "but",
    "by",
    "ca",
    "call",
    "can",
    "cannot",
    "cant",
    "co",
    "con",
    "could",
    "couldnt",
    "cry",
    "de",
    "describe",
    "detail",
    "did",
    "do",
    "does",
    "doing",
    "done",
    "down",
    "due",
    "during",
    "each",
    "eg",
    "eight",
    "either",
    "eleven",
    "else",
    "elsewhere",
    "empty",
    "enough",
    "etc",
    "even",
    "ever",
    "every",
    "everyone",
    "everything",
    "everywhere",
    "except",
    "few",
    "fifteen",
    "fifty",
    "fill",
    "find",
    "fire",
    "first",
    "five",
    "for",
    "former",
    "formerly",
    "forty",
    "found",
    "four",
    "from",
    "front",
    "full",
    "further",
    "get",
    "give",
    "go",
    "had",
    "has",
    "hasnt",
    "have",
    "he",
    "hence",
    "her",
    "here",
    "hereafter",
    "hereby",
    "herein",
    "hereupon",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "however",
    "hundred",
    "i",
    "ie",
    "if",
    "in",
    "inc",
    "indeed",
    "interest",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "keep",
    "last",
    "latter",
    "latterly",
    "least",
    "less",
    "ltd",
    "made",
    "make",
    "many",
    "may",
    "me",
    "meanwhile",
    "might",
    "mill",
    "mine",
    "more",
    "moreover",
    "most",
    "mostly",
    "move",
    "much",
    "must",
    "my",
    "myself",
    "n't",
    "name",
    "namely",
    "neither",
    "never",
    "nevertheless",
    "next",
    "nine",
    "no",
    "nobody",
    "none",
    "noone",
    "nor",
    "not",
    "nothing",
    "now",
    "nowhere",
    "n‘t",
    "n’t",
    "of",
    "off",
    "often",
    "on",
    "once",
    "one",
    "only",
    "onto",
    "or",
    "other",
    "others",
    "otherwise",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "part",
    "per",
    "perhaps",
    "please",
    "put",
    "quite",
    "rather",
    "re",
    "really",
    "regarding",
    "same",
    "say",
    "see",
    "seem",
    "seemed",
    "seeming",
    "seems",
    "serious",
    "several",
    "she",
    "should",
    "show",
    "side",
    "since",
    "sincere",
    "six",
    "sixty",
    "so",
    "some",
    "somehow",
    "someone",
    "something",
    "sometime",
    "sometimes",
    "somewhere",
    "still",
    "such",
    "system",
    "take",
    "ten",
    "than",
    "that",
    "the",
    "their",
    "them",
    "themselves",
    "then",
    "thence",
    "there",
    "thereafter",
    "thereby",
    "therefore",
    "therein",
    "thereupon",
    "these",
    "they",
    "thick",
    "thin",
    "third",
    "this",
    "those",
    "though",
    "three",
    "through",
    "throughout",
    "thru",
    "thus",
    "to",
    "together",
    "too",
    "top",
    "toward",
    "towards",
    "twelve",
    "twenty",
    "two",
    "un",
    "under",
    "unless",
    "until",
    "up",
    "upon",
    "us",
    "used",
    "using",
    "various",
    "very",
    "via",
    "was",
    "we",
    "well",
    "were",
    "what",
    "whatever",
    "when",
    "whence",
    "whenever",
    "where",
    "whereafter",
    "whereas",
    "whereby",
    "wherein",
    "whereupon",
    "wherever",
    "whether",
    "which",
    "while",
    "whither",
    "who",
    "whoever",
    "whole",
    "whom",
    "whose",
    "why",
    "will",
    "with",
    "within",
    "without",
    "would",
    "yet",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "‘d",
    "‘ll",
    "‘m",
    "‘re",
    "‘s",
    "‘ve",
    "’d",
    "’ll",
    "’m",
    "’re",
    "’s",
    "’ve"
}

# =========================
# Text preprocessing / NB
# =========================

def simple_singularize(token):
    if token in ALIASES:
        return ALIASES[token]
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 4 and not token.endswith("ses"):
        return token[:-2]
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token

def tokenize_text(text):
    if pd.isna(text):
        text = "missing"
    text = str(text).lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z'\s]", " ", text)
    tokens = []
    for raw_tok in text.split():
        tok = raw_tok.strip("'")
        if not tok:
            continue
        tok = simple_singularize(tok)
        if len(tok) < MIN_TOKEN_LEN:
            continue
        if tok in stop_words:
            continue
        tokens.append(tok)
    return tokens

def select_top_k_vocab_from_training(texts, k):
    if k <= 0:
        raise ValueError("k must be > 0.")
    doc_counter = {}
    for text in pd.Series(texts).fillna("missing"):
        uniq_tokens = set(tokenize_text(text))
        if EXCLUDE_MISSING_FROM_VOCAB and "missing" in uniq_tokens:
            uniq_tokens.remove("missing")
        for tok in uniq_tokens:
            doc_counter[tok] = doc_counter.get(tok, 0) + 1
    if not doc_counter:
        raise ValueError("No usable tokens found in the training data after preprocessing.")
    items = list(doc_counter.items())
    if SORT_TIES_ALPHABETICALLY:
        items.sort(key=lambda x: (-x[1], x[0]))
    else:
        items.sort(key=lambda x: -x[1])
    vocab = [tok for tok, _ in items[:k]]
    return vocab, doc_counter

class TopKBernoulliNB:
    def __init__(self, k, alpha=NB_ALPHA):
        self.k = int(k)
        self.alpha = float(alpha)
        self.vocab_ = None
        self.vocab_index_ = None
        self.doc_freq_ = None
        self.class_count_ = None
        self.feature_prob_ = None

    def _vectorize_one(self, text):
        row = np.zeros(len(self.vocab_), dtype=np.int8)
        uniq_tokens = set(tokenize_text(text))
        for tok in uniq_tokens:
            j = self.vocab_index_.get(tok)
            if j is not None:
                row[j] = 1
        return row

    def _vectorize(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = pd.Series(texts).fillna("missing")
        X = np.zeros((len(texts), len(self.vocab_)), dtype=np.int8)
        for i, text in enumerate(texts):
            X[i] = self._vectorize_one(text)
        return X

    def fit(self, texts, y):
        self.vocab_, self.doc_freq_ = select_top_k_vocab_from_training(texts, self.k)
        self.vocab_index_ = {tok: i for i, tok in enumerate(self.vocab_)}
        X = self._vectorize(texts)
        y = np.asarray(y, dtype=int)
        K = int(np.max(y)) + 1
        self.class_count_ = np.bincount(y, minlength=K).astype(np.float64)
        V = X.shape[1]
        self.feature_prob_ = np.zeros((K, V), dtype=np.float64)
        for c in range(K):
            mask = (y == c)
            n_c = float(mask.sum())
            count = X[mask].sum(axis=0).astype(np.float64)
            self.feature_prob_[c] = (count + self.alpha) / (n_c + 2.0 * self.alpha)
        self.feature_prob_ = np.clip(self.feature_prob_, 1e-12, 1.0 - 1e-12)
        return self

    def hit_count(self, texts):
        X = self._vectorize(texts)
        return X.sum(axis=1)

    def predict_proba(self, texts):
        X = self._vectorize(texts).astype(np.float64)
        K = len(self.class_count_)
        log_prior = np.log(self.class_count_ / self.class_count_.sum())
        log_p = np.log(self.feature_prob_)
        log_not_p = np.log(1.0 - self.feature_prob_)
        log_scores = X @ log_p.T + (1.0 - X) @ log_not_p.T + log_prior.reshape(1, K)
        if ZERO_HIT_RETURNS_PRIOR:
            hits = X.sum(axis=1)
            zero_mask = hits == 0
            if np.any(zero_mask):
                prior = self.class_count_ / self.class_count_.sum()
                log_scores[zero_mask] = np.log(prior)
        log_scores = log_scores - log_scores.max(axis=1, keepdims=True)
        probs = np.exp(log_scores)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, texts):
        probs = self.predict_proba(texts)
        return np.argmax(probs, axis=1)

    def predict_one(self, text):
        return int(self.predict([text])[0])

def rebuild_nb_model(vocab, class_count, feature_prob, alpha=NB_ALPHA):
    model = TopKBernoulliNB(k=len(vocab), alpha=alpha)
    model.vocab_ = list(vocab)
    model.vocab_index_ = {tok: i for i, tok in enumerate(model.vocab_)}
    model.class_count_ = np.asarray(class_count, dtype=np.float64)
    model.feature_prob_ = np.asarray(feature_prob, dtype=np.float64)
    return model

def load_default_model_from_embedded_params():
    required_names = [
        "W",
        "MU",
        "SIGMA",
        "FEATURE_COLUMNS",
        "FOOD_VOCAB",
        "FOOD_CLASS_COUNT",
        "FOOD_FEATURE_PROB",
        "FEELING_VOCAB",
        "FEELING_CLASS_COUNT",
        "FEELING_FEATURE_PROB",
        "SOUNDTRACK_VOCAB",
        "SOUNDTRACK_CLASS_COUNT",
        "SOUNDTRACK_FEATURE_PROB",
        "MULTIHOT_ENCODERS",
    ]

    for name in required_names:
        if name not in globals():
            raise NameError(f"Missing embedded model parameter: {name}")

    food_model = rebuild_nb_model(
        FOOD_VOCAB,
        FOOD_CLASS_COUNT,
        FOOD_FEATURE_PROB,
    )
    feeling_model = rebuild_nb_model(
        FEELING_VOCAB,
        FEELING_CLASS_COUNT,
        FEELING_FEATURE_PROB,
    )
    soundtrack_model = rebuild_nb_model(
        SOUNDTRACK_VOCAB,
        SOUNDTRACK_CLASS_COUNT,
        SOUNDTRACK_FEATURE_PROB,
    )

    multihot_bundle = MultiHotCategoryBundle(MULTIHOT_COLS)
    multihot_bundle.encoders = MULTIHOT_ENCODERS

    trained = {
        "W": np.asarray(W, dtype=np.float64),
        "mu": np.asarray(MU, dtype=np.float64),
        "sigma": np.asarray(SIGMA, dtype=np.float64),
        "food_model": food_model,
        "feeling_model": feeling_model,
        "soundtrack_model": soundtrack_model,
        "multihot_bundle": multihot_bundle,
        "feature_columns": list(FEATURE_COLUMNS),
    }
    return trained

def train_text_nb_models(df_train, food_k=FOOD_K, feeling_k=FEELING_K, soundtrack_k=SOUNDTRACK_K, alpha=NB_ALPHA):
    food_model = TopKBernoulliNB(k=food_k, alpha=alpha).fit(df_train[FOOD_COL], df_train[LABEL_COL])
    feeling_model = TopKBernoulliNB(k=feeling_k, alpha=alpha).fit(df_train[FEEL_COL], df_train[LABEL_COL])
    soundtrack_model = TopKBernoulliNB(k=soundtrack_k, alpha=alpha).fit(df_train[SOUND_COL], df_train[LABEL_COL])
    return food_model, feeling_model, soundtrack_model

def build_nb_feature_frame(df_any, food_model, feeling_model, soundtrack_model):
    food_probs = food_model.predict_proba(df_any[FOOD_COL].fillna("missing"))
    feeling_probs = feeling_model.predict_proba(df_any[FEEL_COL].fillna("missing"))
    soundtrack_probs = soundtrack_model.predict_proba(df_any[SOUND_COL].fillna("missing"))

    food_hits = food_model.hit_count(df_any[FOOD_COL].fillna("missing"))
    feeling_hits = feeling_model.hit_count(df_any[FEEL_COL].fillna("missing"))
    soundtrack_hits = soundtrack_model.hit_count(df_any[SOUND_COL].fillna("missing"))

    return pd.DataFrame(
        {
            "food_nb_p0": food_probs[:, 0],
            "food_nb_p1": food_probs[:, 1],
            "food_nb_p2": food_probs[:, 2],
            "food_nb_pred": np.argmax(food_probs, axis=1),
            "food_nb_hit_count": food_hits,
            "food_nb_zero_hit": (food_hits == 0).astype(int),
            "feeling_nb_p0": feeling_probs[:, 0],
            "feeling_nb_p1": feeling_probs[:, 1],
            "feeling_nb_p2": feeling_probs[:, 2],
            "feeling_nb_pred": np.argmax(feeling_probs, axis=1),
            "feeling_nb_hit_count": feeling_hits,
            "feeling_nb_zero_hit": (feeling_hits == 0).astype(int),
            "soundtrack_nb_p0": soundtrack_probs[:, 0],
            "soundtrack_nb_p1": soundtrack_probs[:, 1],
            "soundtrack_nb_p2": soundtrack_probs[:, 2],
            "soundtrack_nb_pred": np.argmax(soundtrack_probs, axis=1),
            "soundtrack_nb_hit_count": soundtrack_hits,
            "soundtrack_nb_zero_hit": (soundtrack_hits == 0).astype(int),
        },
        index=df_any.index,
    )

# =========================
# multihot for the 3 categorical variables
# =========================
def split_multiselect_cell(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if (not s) or s.lower() == "missing":
        return []
    return [part.strip() for part in s.split(",") if part.strip() and part.strip().lower() != "missing"]

class MultiHotCategoryBundle:
    def __init__(self, columns):
        self.columns = list(columns)
        self.encoders = {}

    def fit(self, df_train):
        for col in self.columns:
            classes = sorted(set(sum(df_train[col].apply(split_multiselect_cell).tolist(), [])))
            self.encoders[col] = classes
        return self

    def transform(self, df_any):
        frames = []
        for col in self.columns:
            classes = self.encoders[col]
            index_map = {cls: i for i, cls in enumerate(classes)}
            arr = np.zeros((len(df_any), len(classes)), dtype=np.int8)
            rows = df_any[col].apply(split_multiselect_cell)
            for i, items in enumerate(rows):
                for item in items:
                    j = index_map.get(item)
                    if j is not None:
                        arr[i, j] = 1
            safe_col_prefix = re.sub(r"[^0-9a-zA-Z]+", "_", col).strip("_").lower()[:24]
            col_names = [f"{safe_col_prefix}__{re.sub(r'[^0-9a-zA-Z]+', '_', cls).strip('_').lower()}" for cls in classes]
            frames.append(pd.DataFrame(arr, columns=col_names, index=df_any.index))
        return pd.concat(frames, axis=1)

# =========================
# Logistic-regression features
# =========================
def get_base_feature_cols(df_any):
    exclude_cols = {
        LABEL_COL,
        UNIQUE_ID_COL,
        "Painting",
        FOOD_COL,
        FEEL_COL,
        SOUND_COL,
        ROOM_COL,
        WHO_COL,
        SEASON_COL,
    }
    candidate_cols = [col for col in df_any.columns if col not in exclude_cols]
    numeric_cols = []
    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(df_any[col]) or pd.api.types.is_bool_dtype(df_any[col]):
            numeric_cols.append(col)
    return numeric_cols

def build_logistic_feature_matrix(df_any, food_model, feeling_model, soundtrack_model, multihot_bundle):
    base_cols = get_base_feature_cols(df_any)
    base_X = df_any[base_cols].copy()
    multihot_X = multihot_bundle.transform(df_any)
    nb_X = build_nb_feature_frame(df_any, food_model, feeling_model, soundtrack_model)
    X = pd.concat([
        base_X.reset_index(drop=True),
        multihot_X.reset_index(drop=True),
        nb_X.reset_index(drop=True),
    ], axis=1)
    return X

# =========================
# StandardScaler
# =========================
def fit_standardizer(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma[sigma < 1e-12] = 1.0
    return mu, sigma

def apply_standardizer(X, mu, sigma):
    return (X - mu) / sigma

# =========================
# Full-K multinomial logistic regression (Newton)
# =========================
def softmax_full(scores):
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def fit_multinomial_logistic_newton(X, y, C=LOGREG_C, max_iter=NEWTON_MAX_ITER, tol=NEWTON_TOL, intercept_damping=INTERCEPT_DAMPING):
    y = np.asarray(y, dtype=int)
    N, D = X.shape
    K = int(np.max(y)) + 1

    Xb = np.concatenate([np.ones((N, 1), dtype=np.float64), X], axis=1)
    D1 = D + 1

    W = np.zeros((D1, K), dtype=np.float64)
    # Exact sklearn lbfgs scaling for multinomial logistic:
    # l2_reg_strength = 1.0 / (C * sw_sum), with sw_sum = N here.
    lam = 1.0 / (float(C) * float(N))

    reg_diag = np.ones(D1, dtype=np.float64)
    reg_diag[0] = intercept_damping

    Y = np.zeros((N, K), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    def objective(Wcur):
        P = softmax_full(Xb @ Wcur)
        loss = -np.mean(np.log(np.clip(P[np.arange(N), y], 1e-15, 1.0)))
        reg = 0.5 * lam * (np.sum(Wcur[1:, :] ** 2) + intercept_damping * np.sum(Wcur[0:1, :] ** 2))
        return loss + reg

    for _ in range(max_iter):
        P = softmax_full(Xb @ W)

        G = np.zeros((D1, K), dtype=np.float64)
        H = np.zeros((D1 * K, D1 * K), dtype=np.float64)
        Xt = Xb.T

        for a in range(K):
            G[:, a] = (Xt @ (P[:, a] - Y[:, a])) / N + lam * reg_diag * W[:, a]

        for a in range(K):
            pa = P[:, a]
            for b in range(K):
                if a == b:
                    w_ab = pa * (1.0 - pa)
                else:
                    w_ab = -pa * P[:, b]
                H_block = (Xt @ (Xb * w_ab[:, None])) / N
                if a == b:
                    H_block = H_block + lam * np.diag(reg_diag)
                r0 = a * D1
                c0 = b * D1
                H[r0:r0 + D1, c0:c0 + D1] = H_block

        H = H + 1e-8 * np.eye(H.shape[0], dtype=np.float64)
        g = G.reshape(-1, order="F")

        try:
            delta = np.linalg.solve(H, g)
        except Exception:
            delta = np.linalg.lstsq(H, g, rcond=None)[0]

        step = 1.0
        W_new = W - delta.reshape(D1, K, order="F")
        old_obj = objective(W)
        new_obj = objective(W_new)
        while new_obj > old_obj and step > 1e-8:
            step *= 0.5
            W_new = W - step * delta.reshape(D1, K, order="F")
            new_obj = objective(W_new)

        max_change = np.max(np.abs(W_new - W))
        W = W_new
        if max_change < tol:
            break

    return W

def predict_multinomial_logistic(W, X):
    Xb = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), X], axis=1)
    probs = softmax_full(Xb @ W)
    pred = np.argmax(probs, axis=1)
    return pred, probs

# =========================
# Training / prediction wrapper
# =========================
def train_logistic_with_nb_features(
    df_train,
    food_k=FOOD_K,
    feeling_k=FEELING_K,
    soundtrack_k=SOUNDTRACK_K,
    nb_alpha=NB_ALPHA,
    C=LOGREG_C,
):
    food_model, feeling_model, soundtrack_model = train_text_nb_models(
        df_train,
        food_k=food_k,
        feeling_k=feeling_k,
        soundtrack_k=soundtrack_k,
        alpha=nb_alpha,
    )
    multihot_bundle = MultiHotCategoryBundle(MULTIHOT_COLS).fit(df_train)

    X_train = build_logistic_feature_matrix(
        df_train,
        food_model,
        feeling_model,
        soundtrack_model,
        multihot_bundle,
    )
    feature_columns = X_train.columns.tolist()

    y_train = df_train[LABEL_COL].to_numpy(dtype=int)

    mu, sigma = fit_standardizer(X_train.to_numpy(dtype=np.float64))
    X_train_scaled = apply_standardizer(X_train.to_numpy(dtype=np.float64), mu, sigma)

    W = fit_multinomial_logistic_newton(X_train_scaled, y_train, C=C)

    return {
        "W": W,
        "mu": mu,
        "sigma": sigma,
        "food_model": food_model,
        "feeling_model": feeling_model,
        "soundtrack_model": soundtrack_model,
        "multihot_bundle": multihot_bundle,
        "feature_columns": feature_columns,
    }

def predict_logistic_from_models(df_any, trained):
    X_any = build_logistic_feature_matrix(
        df_any,
        trained["food_model"],
        trained["feeling_model"],
        trained["soundtrack_model"],
        trained["multihot_bundle"],
    )

    feature_columns = trained.get("feature_columns")
    if feature_columns is not None:
        X_any = X_any.reindex(columns=feature_columns, fill_value=0)

    X_scaled = apply_standardizer(
        X_any.to_numpy(dtype=np.float64),
        trained["mu"],
        trained["sigma"],
    )
    pred, probs = predict_multinomial_logistic(trained["W"], X_scaled)
    out = df_any.copy()
    out["pred"] = pred
    return out, probs

# =========================
# Exact numpy StratifiedKFold implementation matching sklearn logic
# =========================
def stratified_kfold_indices(y, n_splits=5, shuffle=True, random_state=42):
    y = np.asarray(y)
    # y_encoded: class labels encoded by order of appearance, just like sklearn
    _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
    _, class_perm = np.unique(y_idx, return_inverse=True)
    y_encoded = class_perm[y_inv]

    n_classes = len(y_idx)
    y_order = np.sort(y_encoded)
    allocation = np.asarray([
        np.bincount(y_order[i::n_splits], minlength=n_classes)
        for i in range(n_splits)
    ])

    rng = np.random.RandomState(random_state)
    test_folds = np.empty(len(y), dtype=int)
    for k in range(n_classes):
        folds_for_class = np.arange(n_splits).repeat(allocation[:, k])
        if shuffle:
            rng.shuffle(folds_for_class)
        test_folds[y_encoded == k] = folds_for_class

    splits = []
    all_idx = np.arange(len(y))
    for fold_id in range(n_splits):
        val_idx = np.where(test_folds == fold_id)[0]
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[val_idx] = False
        train_idx = all_idx[train_mask]
        splits.append((train_idx, val_idx))
    return splits

def evaluate_one_param_combo_kfold(
    df,
    food_k=FOOD_K,
    feeling_k=FEELING_K,
    soundtrack_k=SOUNDTRACK_K,
    nb_alpha=NB_ALPHA,
    C=LOGREG_C,
    n_splits=5,
    random_state=42,
):
    y = df[LABEL_COL].to_numpy(dtype=int)
    splits = stratified_kfold_indices(y, n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores = []
    for train_idx, val_idx in splits:
        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()

        trained = train_logistic_with_nb_features(
            df_train=df_train,
            food_k=food_k,
            feeling_k=feeling_k,
            soundtrack_k=soundtrack_k,
            nb_alpha=nb_alpha,
            C=C,
        )
        val_pred_df, _ = predict_logistic_from_models(df_val, trained)
        acc = float((val_pred_df["pred"].to_numpy(dtype=int) == df_val[LABEL_COL].to_numpy(dtype=int)).mean())
        fold_scores.append(acc)

    return {
        "food_k": food_k,
        "feeling_k": feeling_k,
        "soundtrack_k": soundtrack_k,
        "nb_alpha": nb_alpha,
        "C": C,
        "mean_val_acc": float(np.mean(fold_scores)),
        "std_val_acc": float(np.std(fold_scores)),
        "fold_scores": fold_scores,
    }

def train_val_split_indices(y, train_size=0.8, random_state=42, stratify=True):
    y = np.asarray(y)
    n = len(y)
    rng = np.random.RandomState(random_state)

    if not stratify:
        indices = np.arange(n)
        rng.shuffle(indices)
        n_train = int(round(n * train_size))
        train_idx = np.sort(indices[:n_train])
        val_idx = np.sort(indices[n_train:])
        return train_idx, val_idx

    classes, y_encoded = np.unique(y, return_inverse=True)
    train_parts = []
    val_parts = []
    for class_id in range(len(classes)):
        class_idx = np.where(y_encoded == class_id)[0]
        rng.shuffle(class_idx)
        n_train_class = int(round(len(class_idx) * train_size))
        if len(class_idx) >= 2:
            n_train_class = max(1, min(len(class_idx) - 1, n_train_class))
        train_parts.append(class_idx[:n_train_class])
        val_parts.append(class_idx[n_train_class:])

    train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=int)
    val_idx = np.sort(np.concatenate(val_parts)) if val_parts else np.array([], dtype=int)
    return train_idx, val_idx

def evaluate_one_param_combo_holdout(
    df,
    food_k=FOOD_K,
    feeling_k=FEELING_K,
    soundtrack_k=SOUNDTRACK_K,
    nb_alpha=NB_ALPHA,
    C=LOGREG_C,
    train_size=0.8,
    random_state=42,
    stratify=True,
):
    y = df[LABEL_COL].to_numpy(dtype=int)
    train_idx, val_idx = train_val_split_indices(
        y,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify,
    )
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()

    trained = train_logistic_with_nb_features(
        df_train=df_train,
        food_k=food_k,
        feeling_k=feeling_k,
        soundtrack_k=soundtrack_k,
        nb_alpha=nb_alpha,
        C=C,
    )
    val_pred_df, _ = predict_logistic_from_models(df_val, trained)
    acc = float((val_pred_df["pred"].to_numpy(dtype=int) == df_val[LABEL_COL].to_numpy(dtype=int)).mean())
    return {
        "food_k": food_k,
        "feeling_k": feeling_k,
        "soundtrack_k": soundtrack_k,
        "nb_alpha": nb_alpha,
        "C": C,
        "train_size": train_size,
        "random_state": random_state,
        "stratify": stratify,
        "train_rows": int(len(df_train)),
        "val_rows": int(len(df_val)),
        "val_acc": acc,
    }

# =========================
# Public competition API
# =========================
_MODEL_CACHE = None
_TRAIN_INFO_CACHE = None

def _get_trained_model():
    global _MODEL_CACHE, _TRAIN_INFO_CACHE

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    # First choice: use embedded default model parameters
    try:
        _MODEL_CACHE = load_default_model_from_embedded_params()
        _TRAIN_INFO_CACHE = ("embedded_parameters", True, None)
        return _MODEL_CACHE
    except Exception:
        pass

    # Fallback: if embedded params are missing/broken, try training CSV
    try:
        df_train, chosen_path, used_clean_directly = resolve_training_dataframe()
        _TRAIN_INFO_CACHE = (chosen_path, used_clean_directly, len(df_train))
        _MODEL_CACHE = train_logistic_with_nb_features(df_train, C=LOGREG_C)
        return _MODEL_CACHE
    except Exception as e:
        raise RuntimeError(
            "Could not load embedded model parameters, and training CSV could not be loaded either."
        ) from e

def predict(x):
    model = _get_trained_model()
    df_one_raw = pd.DataFrame([x])
    df_one_clean = clean_raw_dataframe(df_one_raw, drop_rows_with_many_missing=False)
    pred_df, _ = predict_logistic_from_models(df_one_clean, model)
    return LABEL_TO_NAME[int(pred_df["pred"].iloc[0])]

def predict_all(filename):
    """
    Read a RAW-format CSV, clean it using the old table-level cleaning rules,
    but keep all rows at prediction time:
      - anomalous values may be capped / coerced to NaN
      - missing values are imputed using the old default fill rules
      - rows are NOT dropped just because they have many missing values
    """
    model = _get_trained_model()
    df_raw = pd.read_csv(filename)
    df_clean = clean_raw_dataframe(df_raw, drop_rows_with_many_missing=False)
    if len(df_clean) == 0:
        return []
    pred_df, _ = predict_logistic_from_models(df_clean, model)
    return [LABEL_TO_NAME[int(y)] for y in pred_df["pred"].tolist()]

def resolve_optional_csv_path(path):
    """
    Try the given path as-is, then relative to the script directory.
    """
    if not path:
        return path

    script_dir_parts = re.split(r"[\/]", __file__)
    script_dir = "" if len(script_dir_parts) <= 1 else "/".join(script_dir_parts[:-1])

    candidate_paths = [path]
    if script_dir:
        candidate_paths.append(script_dir + "/" + path)

    last_exc = None
    for candidate in candidate_paths:
        try:
            pd.read_csv(candidate, nrows=1)
            return candidate
        except Exception as e:
            last_exc = e
            continue

    raise FileNotFoundError(f"Could not load test CSV: {path}") from last_exc

def evaluate_test_csv_accuracy(test_csv_path):
    """
    Read a RAW-format test CSV, clean it with the old table-level cleaning rules,
    keep all rows at prediction time, predict labels, and compare them against
    the labels generated from Painting.

    Returns a dict with:
      - accuracy
      - rows_raw
      - rows_after_cleaning
      - rows_dropped_by_cleaning
      - pred_df
    """
    model = _get_trained_model()
    resolved_path = resolve_optional_csv_path(test_csv_path)
    df_test_raw = pd.read_csv(resolved_path)

    rows_raw = int(len(df_test_raw))
    df_test_clean = clean_raw_dataframe(df_test_raw, drop_rows_with_many_missing=False)
    rows_after_cleaning = int(len(df_test_clean))
    rows_dropped = rows_raw - rows_after_cleaning

    if rows_after_cleaning == 0:
        raise ValueError("No test rows remain after cleaning.")

    pred_df, _ = predict_logistic_from_models(df_test_clean, model)

    if LABEL_COL not in df_test_clean.columns:
        raise ValueError("Ground-truth label is missing after cleaning.")

    y_true = df_test_clean[LABEL_COL].to_numpy(dtype=int)
    y_pred = pred_df["pred"].to_numpy(dtype=int)
    acc = float((y_true == y_pred).mean())

    pred_df = pred_df.copy()
    pred_df["true_label"] = y_true
    pred_df["pred_name"] = pred_df["pred"].map(LABEL_TO_NAME)
    pred_df["true_name"] = pred_df["true_label"].map(LABEL_TO_NAME)

    return {
        "resolved_path": resolved_path,
        "accuracy": acc,
        "rows_raw": rows_raw,
        "rows_after_cleaning": rows_after_cleaning,
        "rows_dropped_by_cleaning": rows_dropped,
        "pred_df": pred_df,
    }

# if __name__ == "__main__":
#
#     x = predict({
#         "unique_id": 2,
#         "Painting": "The Persistence of Memory",
#         "On a scale of 1–10, how intense is the emotion conveyed by the artwork?": 11,
#         "Describe how this painting makes you feel.": "The clocks are burnt on a hot desert, it embodies the melancholy of time passing. The dominant yellow hue intensifies this sensation.",
#         "This art piece makes me feel sombre.": "4 - Agree",
#         "This art piece makes me feel content.": "3 - Neutral/Unsure",
#         "This art piece makes me feel calm.": "2 - Disagree",
#         "This art piece makes me feel uneasy.": "1 - Strongly disagree",
#         "How many prominent colours do you notice in this painting?": 9999999999999999999999999999999999999999999999999999999999999999999999999999999999999,
#         "How many objects caught your eye in the painting?": -222,
#         "How much (in Canadian dollars) would you be willing to pay for this painting?": "99999999999999999999999999999999999999999999999999999999m",
#         "If you could purchase this painting, which room would you put that painting in?": "Bathroom",
#         "If you could view this art in person, who would you want to view it with?": "By yourself",
#         "What season does this art piece remind you of?": "Fall",
#         "If this painting was a food, what would be?": "Fries",
#         "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.": "A country song that contrasts nostalgia for the past with dissatisfaction with the present"
#     })
#     print(x)
#
#     y = predict_all("test.csv")
#     print(y)
#
#     z = predict_all("F:/machine_learning_learn_2/new/original.csv")
#     print(z)
