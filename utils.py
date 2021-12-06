#'data': [[0.45769231, 0.28076923, 0.26153846], [0.13537118, 0.4628821 , 0.40174672], [0.17741935, 0.31451613, 0.50806452]]
#'labels': [race_ASIAN, race_BLACK/AFRICAN AMERICAN, race_WHITE]
def plot_confusion_matrix():
    return

# {'roc_auc_curves': {'Atelectasis': {'fpr': array([0.     , 0.     , 0.0125 , 0.0125 , 0.0375 , 0.0375 , 0.04375,
#           0.04375, 0.0625 , 0.0625 , 0.0875 , 0.0875 , 0.1    , 0.1    ,
#           0.1125 , 0.1125 , 0.11875, 0.11875, 0.125  , 0.125  , 0.1375 ,
#           0.1375 , 0.15625, 0.15625, 0.1625 , 0.1625 , 0.20625, 0.20625,
#           0.2125 , 0.2125 , 0.25625, 0.25625, 0.26875, 0.26875, 0.29375,
#           0.29375, 0.3875 , 0.3875 , 0.46875, 0.46875, 0.525  , 0.525  ,
#           0.5375 , 0.5375 , 0.575  , 0.575  , 0.6125 , 0.6125 , 0.69375,
#           0.69375, 0.71875, 0.71875, 0.875  , 0.875  , 0.88125, 0.88125,
#           0.8875 , 0.8875 , 1.     ]),
#    'thresholds': [1.2955400943756104,
#     0.2955400347709656,
#     0.2648463249206543,
#     0.26470115780830383,
#     0.22706156969070435,
#     0.2165510654449463,
#     0.21328885853290558,
#     0.20778681337833405,
#     0.2026997208595276,
#     0.19276946783065796,
#     0.1881706267595291,
#     0.18731315433979034,
#     0.1842632293701172,
#     0.18209046125411987,
#     0.17984390258789062,
#     0.17633022367954254,
#     0.1726827323436737,
#     0.168942391872406,
#     0.16735000908374786,
#     0.16703025996685028,
#     0.16432441771030426,
#     0.1643134206533432,
#     0.16274434328079224,
#     0.1622970551252365,
#     0.16178077459335327,
#     0.159433051943779,
#     0.15554699301719666,
#     0.15459196269512177,
#     0.15425758063793182,
#     0.15399566292762756,
#     0.14402827620506287,
#     0.14385847747325897,
#     0.1412581205368042,
#     0.1404554843902588,
#     0.13538776338100433,
#     0.1351107358932495,
#     0.12335579842329025,
#     0.11896288394927979,
#     0.10648970305919647,
#     0.10456109791994095,
#     0.09916338324546814,
#     0.09876859188079834,
#     0.09795350581407547,
#     0.0955321192741394,
#     0.09266099333763123,
#     0.09242323786020279,
#     0.08676028251647949,
#     0.08554770797491074,
#     0.07922700047492981,
#     0.07890944927930832,
#     0.07716292887926102,
#     0.07657304406166077,
#     0.05999958887696266,
#     0.05890599265694618,
#     0.05858680605888367,
#     0.058366693556308746,
#     0.057180460542440414,
#     0.05658326670527458,
#     0.02368723228573799],
#    'tpr': array([0.   , 0.025, 0.025, 0.05 , 0.05 , 0.125, 0.125, 0.2  , 0.2  ,
#           0.325, 0.325, 0.35 , 0.35 , 0.4  , 0.4  , 0.425, 0.425, 0.45 ,
#           0.45 , 0.475, 0.475, 0.5  , 0.5  , 0.525, 0.525, 0.55 , 0.55 ,
#           0.575, 0.575, 0.6  , 0.6  , 0.625, 0.625, 0.65 , 0.65 , 0.675,
#           0.675, 0.7  , 0.7  , 0.725, 0.725, 0.75 , 0.75 , 0.8  , 0.8  ,
#           0.825, 0.825, 0.85 , 0.85 , 0.9  , 0.9  , 0.925, 0.925, 0.95 ,
#           0.95 , 0.975, 0.975, 1.   , 1.   ])},
#   'Cardiomegaly': {'fpr': array([0.        , 0.        , 0.        , 0.02777778, 0.02777778,
#           0.04444444, 0.04444444, 0.05      , 0.05      , 0.06111111,
#           0.06111111, 0.13888889, 0.13888889, 0.15555556, 0.15555556,
#           0.17222222, 0.17222222, 0.21666667, 0.21666667, 0.22222222,
#           0.22222222, 0.23333333, 0.23333333, 0.31111111, 0.31111111,
#           0.31666667, 0.31666667, 0.32777778, 0.32777778, 0.39444444,
#           0.39444444, 0.73333333, 0.73333333, 0.8       , 0.8       ,
#           0.90555556, 0.90555556, 0.93888889, 0.93888889, 1.        ]),
#    'thresholds': [1.79877507686615,
#     0.7987750768661499,
#     0.6378878355026245,
#     0.4406372904777527,
#     0.4111109972000122,
#     0.36693981289863586,
#     0.3661392033100128,
#     0.3627576529979706,
#     0.3456411361694336,
#     0.3383474349975586,
#     0.3045681416988373,
#     0.2100396752357483,
#     0.20800454914569855,
#     0.202874556183815,
#     0.2020668089389801,
#     0.19743047654628754,
#     0.1957838088274002,
#     0.17794860899448395,
#     0.17718981206417084,
#     0.17699460685253143,
#     0.17625121772289276,
#     0.1742846667766571,
#     0.17408928275108337,
#     0.15282632410526276,
#     0.15196900069713593,
#     0.14936843514442444,
#     0.14124006032943726,
#     0.13685747981071472,
#     0.1346379667520523,
#     0.11742141842842102,
#     0.11649534851312637,
#     0.05506501719355583,
#     0.054866403341293335,
#     0.0461978055536747,
#     0.04445653408765793,
#     0.030795574188232422,
#     0.0300718042999506,
#     0.02425987273454666,
#     0.02403886429965496,
#     0.009089219383895397],
#    'tpr': array([0.  , 0.05, 0.1 , 0.1 , 0.15, 0.15, 0.2 , 0.2 , 0.25, 0.25, 0.3 ,
#           0.3 , 0.35, 0.35, 0.4 , 0.4 , 0.45, 0.45, 0.5 , 0.5 , 0.55, 0.55,
#           0.6 , 0.6 , 0.65, 0.65, 0.7 , 0.7 , 0.75, 0.75, 0.8 , 0.8 , 0.85,
#           0.85, 0.9 , 0.9 , 0.95, 0.95, 1.  , 1.  ])},
#   'Consolidation': {'fpr': array([0.        , 0.00531915, 0.07446809, 0.07446809, 0.09574468,
#           0.09574468, 0.13297872, 0.13297872, 0.16489362, 0.16489362,
#           0.17021277, 0.17021277, 0.20744681, 0.20744681, 0.22340426,
#           0.22340426, 0.40957447, 0.40957447, 0.60106383, 0.60106383,
#           0.65425532, 0.65425532, 0.81914894, 0.81914894, 0.93085106,
#           0.93085106, 1.        ]),
#    'thresholds': [1.1747534275054932,
#     0.17475344240665436,
#     0.08842210471630096,
#     0.08819073438644409,
#     0.08212912827730179,
#     0.07755587249994278,
#     0.06558065116405487,
#     0.06424389034509659,
#     0.06160404533147812,
#     0.061505626887083054,
#     0.06132972985506058,
#     0.058997612446546555,
#     0.05676241219043732,
#     0.05647075176239014,
#     0.05236301198601723,
#     0.051021624356508255,
#     0.041528306901454926,
#     0.040725160390138626,
#     0.03051113709807396,
#     0.030430525541305542,
#     0.028431028127670288,
#     0.02834147959947586,
#     0.022089282050728798,
#     0.021809978410601616,
#     0.013763257302343845,
#     0.013435829430818558,
#     0.004429569933563471],
#    'tpr': array([0.        , 0.        , 0.        , 0.08333333, 0.08333333,
#           0.16666667, 0.16666667, 0.25      , 0.25      , 0.33333333,
#           0.33333333, 0.41666667, 0.41666667, 0.5       , 0.5       ,
#           0.58333333, 0.58333333, 0.66666667, 0.66666667, 0.75      ,
#           0.75      , 0.83333333, 0.83333333, 0.91666667, 0.91666667,
#           1.        , 1.        ])},
#   'Edema': {'fpr': array([0.        , 0.        , 0.01369863, 0.01369863, 0.02739726,
#           0.02739726, 0.04109589, 0.04109589, 0.06849315, 0.06849315,
#           0.08219178, 0.08219178, 0.0890411 , 0.0890411 , 0.11643836,
#           0.11643836, 0.12328767, 0.12328767, 0.14383562, 0.14383562,
#           0.15068493, 0.15068493, 0.17808219, 0.17808219, 0.20547945,
#           0.20547945, 0.21232877, 0.21232877, 0.21917808, 0.21917808,
#           0.23287671, 0.23287671, 0.24657534, 0.24657534, 0.26712329,
#           0.26712329, 0.2739726 , 0.2739726 , 0.32876712, 0.32876712,
#           0.33561644, 0.33561644, 0.3630137 , 0.3630137 , 0.38356164,
#           0.38356164, 0.41780822, 0.41780822, 0.43835616, 0.43835616,
#           0.44520548, 0.44520548, 0.51369863, 0.51369863, 0.53424658,
#           0.53424658, 0.60958904, 0.60958904, 0.61643836, 0.61643836,
#           0.63013699, 0.63013699, 0.65068493, 0.65068493, 0.65753425,
#           0.65753425, 0.71232877, 0.71232877, 0.74657534, 0.74657534,
#           0.9109589 , 0.9109589 , 1.        ]),
#    'thresholds': [1.8148245811462402,
#     0.814824640750885,
#     0.7186622023582458,
#     0.6022443175315857,
#     0.5850376486778259,
#     0.5547850728034973,
#     0.503220796585083,
#     0.48444125056266785,
#     0.47618746757507324,
#     0.45534980297088623,
#     0.44625040888786316,
#     0.43578872084617615,
#     0.43475642800331116,
#     0.4174712598323822,
#     0.4024755656719208,
#     0.38596484065055847,
#     0.385374516248703,
#     0.3760402500629425,
#     0.37279024720191956,
#     0.36207371950149536,
#     0.3502695858478546,
#     0.347195029258728,
#     0.33286479115486145,
#     0.32119259238243103,
#     0.3036416172981262,
#     0.282714307308197,
#     0.282043993473053,
#     0.2782946825027466,
#     0.27161553502082825,
#     0.26643067598342896,
#     0.25827059149742126,
#     0.2577214241027832,
#     0.2554401755332947,
#     0.25289881229400635,
#     0.24686852097511292,
#     0.2340332716703415,
#     0.23194779455661774,
#     0.22980143129825592,
#     0.21321699023246765,
#     0.21045662462711334,
#     0.2102539986371994,
#     0.20947107672691345,
#     0.2025637924671173,
#     0.2012903392314911,
#     0.19520848989486694,
#     0.19340960681438446,
#     0.18606479465961456,
#     0.1845066100358963,
#     0.18086659908294678,
#     0.17989058792591095,
#     0.17873775959014893,
#     0.17578762769699097,
#     0.16156412661075592,
#     0.1580711156129837,
#     0.14986011385917664,
#     0.1439974308013916,
#     0.1323230266571045,
#     0.1303134262561798,
#     0.12977424263954163,
#     0.12828341126441956,
#     0.12403842806816101,
#     0.12384136766195297,
#     0.11668865382671356,
#     0.11576561629772186,
#     0.11517854034900665,
#     0.10712588578462601,
#     0.08919661492109299,
#     0.08854132890701294,
#     0.07699673622846603,
#     0.07539568096399307,
#     0.043703652918338776,
#     0.04231400787830353,
#     0.014107988215982914],
#    'tpr': array([0.        , 0.01851852, 0.01851852, 0.09259259, 0.09259259,
#           0.14814815, 0.14814815, 0.18518519, 0.18518519, 0.22222222,
#           0.22222222, 0.27777778, 0.27777778, 0.31481481, 0.31481481,
#           0.33333333, 0.33333333, 0.37037037, 0.37037037, 0.38888889,
#           0.38888889, 0.40740741, 0.40740741, 0.42592593, 0.42592593,
#           0.46296296, 0.46296296, 0.5       , 0.5       , 0.53703704,
#           0.53703704, 0.55555556, 0.55555556, 0.57407407, 0.57407407,
#           0.62962963, 0.62962963, 0.66666667, 0.66666667, 0.7037037 ,
#           0.7037037 , 0.72222222, 0.72222222, 0.74074074, 0.74074074,
#           0.75925926, 0.75925926, 0.77777778, 0.77777778, 0.7962963 ,
#           0.7962963 , 0.81481481, 0.81481481, 0.83333333, 0.83333333,
#           0.85185185, 0.85185185, 0.87037037, 0.87037037, 0.88888889,
#           0.88888889, 0.90740741, 0.90740741, 0.92592593, 0.92592593,
#           0.94444444, 0.94444444, 0.96296296, 0.96296296, 0.98148148,
#           0.98148148, 1.        , 1.        ])},
#   'Enlarged Cardiomediastinum': {'fpr': array([0.        , 0.00534759, 0.00534759, 0.01069519, 0.01069519,
#           0.02673797, 0.02673797, 0.03208556, 0.03208556, 0.15508021,
#           0.15508021, 0.26203209, 0.26203209, 0.28342246, 0.28342246,
#           0.34224599, 0.34224599, 0.49197861, 0.49197861, 0.54010695,
#           0.54010695, 0.55080214, 0.55080214, 0.57754011, 0.57754011,
#           0.87165775, 0.87165775, 1.        ]),
#    'thresholds': [1.2825596332550049,
#     0.28255969285964966,
#     0.2653878927230835,
#     0.24814428389072418,
#     0.2149999886751175,
#     0.2055630385875702,
#     0.20318496227264404,
#     0.19654515385627747,
#     0.19550862908363342,
#     0.12320351600646973,
#     0.1230192556977272,
#     0.10261765122413635,
#     0.10174402594566345,
#     0.10007604956626892,
#     0.09907151758670807,
#     0.09223823994398117,
#     0.08701631426811218,
#     0.07166014611721039,
#     0.0715557262301445,
#     0.06844481080770493,
#     0.06825722008943558,
#     0.0677369087934494,
#     0.06724195927381516,
#     0.06538417190313339,
#     0.06502502411603928,
#     0.04285478591918945,
#     0.04238300025463104,
#     0.019113287329673767],
#    'tpr': array([0.        , 0.        , 0.07692308, 0.07692308, 0.15384615,
#           0.15384615, 0.23076923, 0.23076923, 0.30769231, 0.30769231,
#           0.38461538, 0.38461538, 0.46153846, 0.46153846, 0.53846154,
#           0.53846154, 0.61538462, 0.61538462, 0.69230769, 0.69230769,
#           0.76923077, 0.76923077, 0.84615385, 0.84615385, 0.92307692,
#           0.92307692, 1.        , 1.        ])},
#   'Fracture': {'fpr': array([0.        , 0.00529101, 0.01587302, 0.01587302, 0.08994709,
#           0.08994709, 0.20634921, 0.20634921, 0.38095238, 0.38095238,
#           0.4021164 , 0.4021164 , 0.45502646, 0.45502646, 0.61375661,
#           0.61375661, 0.62962963, 0.62962963, 0.72486772, 0.72486772,
#           0.85185185, 0.85185185, 0.87830688, 0.87830688, 1.        ]),
#    'thresholds': [1.1312119960784912,
#     0.13121198117733002,
#     0.06963112950325012,
#     0.06484746187925339,
#     0.039701711386442184,
#     0.039493776857852936,
#     0.028759349137544632,
#     0.02866225689649582,
#     0.022079993039369583,
#     0.021972397342324257,
#     0.0216403566300869,
#     0.02162482775747776,
#     0.020005034282803535,
#     0.019581731408834457,
#     0.01486336998641491,
#     0.014794043265283108,
#     0.014579770155251026,
#     0.01457331795245409,
#     0.012050287798047066,
#     0.01161936204880476,
#     0.008682467974722385,
#     0.00844605453312397,
#     0.0075331819243729115,
#     0.007452409714460373,
#     0.002927056048065424],
#    'tpr': array([0.        , 0.        , 0.        , 0.09090909, 0.09090909,
#           0.18181818, 0.18181818, 0.27272727, 0.27272727, 0.36363636,
#           0.36363636, 0.45454545, 0.45454545, 0.54545455, 0.54545455,
#           0.63636364, 0.63636364, 0.72727273, 0.72727273, 0.81818182,
#           0.81818182, 0.90909091, 0.90909091, 1.        , 1.        ])},
#   'Lung Lesion': {'fpr': array([0.        , 0.00531915, 0.0106383 , 0.0106383 , 0.04787234,
#           0.04787234, 0.05851064, 0.05851064, 0.06914894, 0.06914894,
#           0.07978723, 0.07978723, 0.11702128, 0.11702128, 0.21276596,
#           0.21276596, 0.23404255, 0.23404255, 0.24468085, 0.24468085,
#           0.65425532, 0.65425532, 0.95212766, 0.95212766, 1.        ]),
#    'thresholds': [1.1984577178955078,
#     0.1984577476978302,
#     0.16215266287326813,
#     0.15299810469150543,
#     0.10208943486213684,
#     0.10097729414701462,
#     0.09283386915922165,
#     0.0904124453663826,
#     0.08772280067205429,
#     0.08315820246934891,
#     0.08017923682928085,
#     0.07967592775821686,
#     0.06811151653528214,
#     0.06748973578214645,
#     0.0505034439265728,
#     0.05035753548145294,
#     0.048132333904504776,
#     0.04809078201651573,
#     0.04728887230157852,
#     0.04680231958627701,
#     0.019137337803840637,
#     0.019053248688578606,
#     0.006363383959978819,
#     0.006199817173182964,
#     0.002602406544610858],
#    'tpr': array([0.        , 0.        , 0.        , 0.08333333, 0.08333333,
#           0.16666667, 0.16666667, 0.25      , 0.25      , 0.33333333,
#           0.33333333, 0.41666667, 0.41666667, 0.5       , 0.5       ,
#           0.58333333, 0.58333333, 0.66666667, 0.66666667, 0.75      ,
#           0.75      , 0.91666667, 0.91666667, 1.        , 1.        ])},
#   'Lung Opacity': {'fpr': array([0.        , 0.        , 0.        , 0.01020408, 0.01020408,
#           0.02040816, 0.02040816, 0.04081633, 0.04081633, 0.05102041,
#           0.05102041, 0.06122449, 0.06122449, 0.07142857, 0.07142857,
#           0.09183673, 0.09183673, 0.10204082, 0.10204082, 0.12244898,
#           0.12244898, 0.13265306, 0.13265306, 0.14285714, 0.14285714,
#           0.15306122, 0.15306122, 0.19387755, 0.19387755, 0.20408163,
#           0.20408163, 0.21428571, 0.21428571, 0.2244898 , 0.2244898 ,
#           0.24489796, 0.24489796, 0.29591837, 0.29591837, 0.30612245,
#           0.30612245, 0.32653061, 0.32653061, 0.34693878, 0.34693878,
#           0.36734694, 0.36734694, 0.37755102, 0.37755102, 0.39795918,
#           0.39795918, 0.41836735, 0.41836735, 0.43877551, 0.43877551,
#           0.44897959, 0.44897959, 0.47959184, 0.47959184, 0.5       ,
#           0.5       , 0.52040816, 0.52040816, 0.55102041, 0.55102041,
#           0.57142857, 0.57142857, 0.58163265, 0.58163265, 0.62244898,
#           0.62244898, 0.63265306, 0.63265306, 0.66326531, 0.66326531,
#           0.73469388, 0.73469388, 0.74489796, 0.74489796, 0.76530612,
#           0.76530612, 0.7755102 , 0.7755102 , 0.84693878, 0.84693878,
#           0.85714286, 0.85714286, 0.8877551 , 0.8877551 , 0.93877551,
#           0.93877551, 0.98979592, 0.98979592, 1.        ]),
#    'thresholds': [1.8197827339172363,
#     0.8197826743125916,
#     0.7920933961868286,
#     0.7912072539329529,
#     0.780875563621521,
#     0.7755062580108643,
#     0.7454653382301331,
#     0.7255688309669495,
#     0.7198439836502075,
#     0.7194952368736267,
#     0.6742563843727112,
#     0.6726433634757996,
#     0.6692709922790527,
#     0.6688770651817322,
#     0.6681044101715088,
#     0.6560594439506531,
#     0.6261457800865173,
#     0.6235346794128418,
#     0.6166936159133911,
#     0.6113086938858032,
#     0.6082867383956909,
#     0.6045657992362976,
#     0.5951002240180969,
#     0.5910829901695251,
#     0.5908500552177429,
#     0.5898848176002502,
#     0.5883769989013672,
#     0.5833828449249268,
#     0.5722192525863647,
#     0.5709110498428345,
#     0.5699724555015564,
#     0.5674763321876526,
#     0.5576151609420776,
#     0.5554935932159424,
#     0.5512779355049133,
#     0.5488205552101135,
#     0.545624852180481,
#     0.5404948592185974,
#     0.5380947589874268,
#     0.5346572399139404,
#     0.5344792008399963,
#     0.5337924361228943,
#     0.5283689498901367,
#     0.5228046774864197,
#     0.5175734162330627,
#     0.5155690312385559,
#     0.5105887651443481,
#     0.5105676651000977,
#     0.5093189477920532,
#     0.5058313608169556,
#     0.5015596747398376,
#     0.4991697072982788,
#     0.4971674680709839,
#     0.493403822183609,
#     0.47900962829589844,
#     0.47840917110443115,
#     0.4753071367740631,
#     0.45945900678634644,
#     0.4542192220687866,
#     0.4529745578765869,
#     0.45201000571250916,
#     0.4262688159942627,
#     0.4242824912071228,
#     0.4173400402069092,
#     0.4121650159358978,
#     0.41109681129455566,
#     0.4097422957420349,
#     0.4084443747997284,
#     0.4081333577632904,
#     0.3928661346435547,
#     0.38759860396385193,
#     0.3859369456768036,
#     0.3786378502845764,
#     0.36631739139556885,
#     0.36410149931907654,
#     0.34435921907424927,
#     0.3399420380592346,
#     0.33747217059135437,
#     0.3370856046676636,
#     0.3250715136528015,
#     0.3205448091030121,
#     0.31885671615600586,
#     0.3180654048919678,
#     0.30229589343070984,
#     0.3004300892353058,
#     0.2986186146736145,
#     0.28988826274871826,
#     0.2741703987121582,
#     0.2728978395462036,
#     0.24026119709014893,
#     0.2368248552083969,
#     0.18200735747814178,
#     0.15520669519901276,
#     0.11478033661842346],
#    'tpr': array([0.        , 0.00980392, 0.01960784, 0.01960784, 0.03921569,
#           0.03921569, 0.04901961, 0.04901961, 0.05882353, 0.05882353,
#           0.12745098, 0.12745098, 0.1372549 , 0.1372549 , 0.14705882,
#           0.14705882, 0.19607843, 0.19607843, 0.23529412, 0.23529412,
#           0.24509804, 0.24509804, 0.2745098 , 0.2745098 , 0.28431373,
#           0.28431373, 0.31372549, 0.31372549, 0.34313725, 0.34313725,
#           0.35294118, 0.35294118, 0.38235294, 0.38235294, 0.42156863,
#           0.42156863, 0.44117647, 0.44117647, 0.45098039, 0.45098039,
#           0.46078431, 0.46078431, 0.48039216, 0.48039216, 0.52941176,
#           0.52941176, 0.55882353, 0.55882353, 0.56862745, 0.56862745,
#           0.59803922, 0.59803922, 0.60784314, 0.60784314, 0.66666667,
#           0.66666667, 0.69607843, 0.69607843, 0.74509804, 0.74509804,
#           0.76470588, 0.76470588, 0.78431373, 0.78431373, 0.80392157,
#           0.80392157, 0.81372549, 0.81372549, 0.82352941, 0.82352941,
#           0.84313725, 0.84313725, 0.85294118, 0.85294118, 0.8627451 ,
#           0.8627451 , 0.89215686, 0.89215686, 0.90196078, 0.90196078,
#           0.92156863, 0.92156863, 0.94117647, 0.94117647, 0.95098039,
#           0.95098039, 0.96078431, 0.96078431, 0.97058824, 0.97058824,
#           0.99019608, 0.99019608, 1.        , 1.        ])},
#   'No Finding': {'fpr': array([0.        , 0.00549451, 0.00549451, 0.01098901, 0.01098901,
#           0.04945055, 0.04945055, 0.05494505, 0.05494505, 0.10989011,
#           0.10989011, 0.21978022, 0.21978022, 0.22527473, 0.22527473,
#           0.25824176, 0.25824176, 0.26923077, 0.26923077, 0.31868132,
#           0.31868132, 0.35164835, 0.35164835, 0.5989011 , 0.5989011 ,
#           0.72527473, 0.72527473, 0.79120879, 0.79120879, 0.81318681,
#           0.81318681, 1.        ]),
#    'thresholds': [1.7642695903778076,
#     0.7642695903778076,
#     0.6490952968597412,
#     0.6374813318252563,
#     0.5900437831878662,
#     0.3579361140727997,
#     0.3444141447544098,
#     0.32950979471206665,
#     0.32265597581863403,
#     0.25517529249191284,
#     0.2486836314201355,
#     0.17928947508335114,
#     0.17875944077968597,
#     0.16839204728603363,
#     0.16649171710014343,
#     0.15519216656684875,
#     0.15402348339557648,
#     0.14883312582969666,
#     0.14837771654129028,
#     0.1331854611635208,
#     0.12778551876544952,
#     0.12245739996433258,
#     0.1216013953089714,
#     0.05703642964363098,
#     0.056371573358774185,
#     0.041940607130527496,
#     0.03882477805018425,
#     0.03102077730000019,
#     0.030910728499293327,
#     0.0285017266869545,
#     0.027438418939709663,
#     0.004532250575721264],
#    'tpr': array([0.        , 0.        , 0.16666667, 0.16666667, 0.22222222,
#           0.22222222, 0.27777778, 0.27777778, 0.33333333, 0.33333333,
#           0.38888889, 0.38888889, 0.44444444, 0.44444444, 0.5       ,
#           0.5       , 0.55555556, 0.55555556, 0.61111111, 0.61111111,
#           0.72222222, 0.72222222, 0.77777778, 0.77777778, 0.83333333,
#           0.83333333, 0.88888889, 0.88888889, 0.94444444, 0.94444444,
#           1.        , 1.        ])},
#   'Pleural Effusion': {'fpr': array([0.        , 0.        , 0.00740741, 0.00740741, 0.02222222,
#           0.02222222, 0.02962963, 0.02962963, 0.03703704, 0.03703704,
#           0.04444444, 0.04444444, 0.05925926, 0.05925926, 0.06666667,
#           0.06666667, 0.07407407, 0.07407407, 0.08148148, 0.08148148,
#           0.0962963 , 0.0962963 , 0.11111111, 0.11111111, 0.11851852,
#           0.11851852, 0.15555556, 0.15555556, 0.17777778, 0.17777778,
#           0.19259259, 0.19259259, 0.22962963, 0.22962963, 0.26666667,
#           0.26666667, 0.27407407, 0.27407407, 0.28148148, 0.28148148,
#           0.28888889, 0.28888889, 0.2962963 , 0.2962963 , 0.31111111,
#           0.31111111, 0.34814815, 0.34814815, 0.36296296, 0.36296296,
#           0.37777778, 0.37777778, 0.40740741, 0.40740741, 0.43703704,
#           0.43703704, 0.48888889, 0.48888889, 0.51851852, 0.51851852,
#           0.52592593, 0.52592593, 0.54074074, 0.54074074, 0.6       ,
#           0.6       , 0.60740741, 0.60740741, 0.65925926, 0.65925926,
#           0.68148148, 0.68148148, 0.77777778, 0.77777778, 0.84444444,
#           0.84444444, 0.91111111, 0.91111111, 1.        ]),
#    'thresholds': [1.962744116783142,
#     0.9627441167831421,
#     0.936948299407959,
#     0.9253720641136169,
#     0.8929669260978699,
#     0.8768090009689331,
#     0.8620259165763855,
#     0.8579400777816772,
#     0.8508062958717346,
#     0.8347150087356567,
#     0.820389986038208,
#     0.8197687268257141,
#     0.8119858503341675,
#     0.8027449250221252,
#     0.7990872859954834,
#     0.7963101267814636,
#     0.7930343747138977,
#     0.780255913734436,
#     0.7655259966850281,
#     0.7640290260314941,
#     0.7617693543434143,
#     0.7551699876785278,
#     0.7389118671417236,
#     0.7382468581199646,
#     0.738142192363739,
#     0.7281312942504883,
#     0.6910479068756104,
#     0.6705822348594666,
#     0.661422610282898,
#     0.6555436253547668,
#     0.6444560289382935,
#     0.6430185437202454,
#     0.6138253808021545,
#     0.6048240065574646,
#     0.5903319716453552,
#     0.5763722658157349,
#     0.575802743434906,
#     0.5735074281692505,
#     0.566429615020752,
#     0.5655193328857422,
#     0.56148362159729,
#     0.5605271458625793,
#     0.5525828003883362,
#     0.552578330039978,
#     0.5326712727546692,
#     0.5296196937561035,
#     0.5103322267532349,
#     0.502183198928833,
#     0.48996052145957947,
#     0.4888908863067627,
#     0.48232588171958923,
#     0.4529798924922943,
#     0.43636956810951233,
#     0.4345429241657257,
#     0.4266582429409027,
#     0.42586448788642883,
#     0.4041372537612915,
#     0.39983251690864563,
#     0.39597249031066895,
#     0.39250826835632324,
#     0.39002805948257446,
#     0.38726645708084106,
#     0.3852347433567047,
#     0.38309189677238464,
#     0.34806492924690247,
#     0.3469001352787018,
#     0.34659209847450256,
#     0.3417224884033203,
#     0.31994086503982544,
#     0.30580294132232666,
#     0.29973000288009644,
#     0.28850555419921875,
#     0.2341868281364441,
#     0.226508229970932,
#     0.17812058329582214,
#     0.17681628465652466,
#     0.1180000975728035,
#     0.11042159795761108,
#     0.03206473961472511],
#    'tpr': array([0.        , 0.01538462, 0.01538462, 0.04615385, 0.04615385,
#           0.06153846, 0.06153846, 0.07692308, 0.07692308, 0.10769231,
#           0.10769231, 0.12307692, 0.12307692, 0.15384615, 0.15384615,
#           0.16923077, 0.16923077, 0.26153846, 0.26153846, 0.27692308,
#           0.27692308, 0.30769231, 0.30769231, 0.33846154, 0.33846154,
#           0.35384615, 0.35384615, 0.49230769, 0.49230769, 0.50769231,
#           0.50769231, 0.52307692, 0.52307692, 0.55384615, 0.55384615,
#           0.6       , 0.6       , 0.63076923, 0.63076923, 0.64615385,
#           0.64615385, 0.66153846, 0.66153846, 0.67692308, 0.67692308,
#           0.69230769, 0.69230769, 0.70769231, 0.70769231, 0.72307692,
#           0.72307692, 0.75384615, 0.75384615, 0.76923077, 0.76923077,
#           0.78461538, 0.78461538, 0.8       , 0.8       , 0.83076923,
#           0.83076923, 0.84615385, 0.84615385, 0.86153846, 0.86153846,
#           0.87692308, 0.87692308, 0.89230769, 0.89230769, 0.92307692,
#           0.92307692, 0.93846154, 0.93846154, 0.96923077, 0.96923077,
#           0.98461538, 0.98461538, 1.        , 1.        ])},
#   'Pleural Other': {'fpr': array([0.        , 0.00502513, 0.12060302, 0.12060302, 1.        ]),
#    'thresholds': [1.0608627796173096,
#     0.06086283177137375,
#     0.007836200296878815,
#     0.007675522938370705,
#     0.00018645086674951017],
#    'tpr': array([0., 0., 0., 1., 1.])},
#   'Pneumonia': {'fpr': array([0.        , 0.00512821, 0.06666667, 0.06666667, 0.14358974,
#           0.14358974, 0.17948718, 0.17948718, 0.46666667, 0.46666667,
#           0.96923077, 0.96923077, 1.        ]),
#    'thresholds': [1.0828903913497925,
#     0.08289042115211487,
#     0.028467388823628426,
#     0.027218028903007507,
#     0.019838161766529083,
#     0.01909901201725006,
#     0.017864100635051727,
#     0.016922440379858017,
#     0.009414803236722946,
#     0.009374713525176048,
#     0.0023326477967202663,
#     0.00226807314902544,
#     0.0014828488929197192],
#    'tpr': array([0. , 0. , 0. , 0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1. , 1. ])},
#   'Pneumothorax': {'fpr': array([0.        , 0.00561798, 0.01123596, 0.01123596, 0.02808989,
#           0.02808989, 0.08988764, 0.08988764, 0.10674157, 0.10674157,
#           0.12921348, 0.12921348, 0.17977528, 0.17977528, 0.19101124,
#           0.19101124, 0.2247191 , 0.2247191 , 0.30337079, 0.30337079,
#           0.5       , 0.5       , 0.51123596, 0.51123596, 0.62359551,
#           0.62359551, 0.74719101, 0.74719101, 0.78651685, 0.78651685,
#           0.81460674, 0.81460674, 0.83146067, 0.83146067, 0.84831461,
#           0.84831461, 0.85393258, 0.85393258, 0.91011236, 0.91011236,
#           0.96067416, 0.96067416, 1.        ]),
#    'thresholds': [1.5019819736480713,
#     0.5019820332527161,
#     0.46618133783340454,
#     0.45142045617103577,
#     0.3989509344100952,
#     0.34867531061172485,
#     0.25309714674949646,
#     0.24630215764045715,
#     0.24036070704460144,
#     0.22433365881443024,
#     0.1974674016237259,
#     0.18144305050373077,
#     0.1439959853887558,
#     0.14069698750972748,
#     0.13892735540866852,
#     0.13684247434139252,
#     0.12772643566131592,
#     0.1273626834154129,
#     0.1090288832783699,
#     0.1086716577410698,
#     0.07445809990167618,
#     0.07290513813495636,
#     0.0693313330411911,
#     0.06716647744178772,
#     0.04811635613441467,
#     0.047652650624513626,
#     0.03636040538549423,
#     0.0363004244863987,
#     0.03067266196012497,
#     0.03063150681555271,
#     0.027597859501838684,
#     0.027581138536334038,
#     0.024746660143136978,
#     0.023864485323429108,
#     0.023055870085954666,
#     0.022906217724084854,
#     0.022548073902726173,
#     0.021952727809548378,
#     0.017179984599351883,
#     0.01688624918460846,
#     0.013944022357463837,
#     0.013851260766386986,
#     0.0033206043299287558],
#    'tpr': array([0.        , 0.        , 0.        , 0.04545455, 0.04545455,
#           0.09090909, 0.09090909, 0.13636364, 0.13636364, 0.22727273,
#           0.22727273, 0.31818182, 0.31818182, 0.36363636, 0.36363636,
#           0.40909091, 0.40909091, 0.45454545, 0.45454545, 0.5       ,
#           0.5       , 0.54545455, 0.54545455, 0.59090909, 0.59090909,
#           0.63636364, 0.63636364, 0.68181818, 0.68181818, 0.72727273,
#           0.72727273, 0.77272727, 0.77272727, 0.81818182, 0.81818182,
#           0.86363636, 0.86363636, 0.90909091, 0.90909091, 0.95454545,
#           0.95454545, 1.        , 1.        ])},
#   'Support Devices': {'fpr': array([0.  , 0.  , 0.  , 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04,
#           0.05, 0.05, 0.06, 0.06, 0.07, 0.07, 0.08, 0.08, 0.09, 0.09, 0.1 ,
#           0.1 , 0.11, 0.11, 0.12, 0.12, 0.13, 0.13, 0.14, 0.14, 0.15, 0.15,
#           0.16, 0.16, 0.17, 0.17, 0.18, 0.18, 0.19, 0.19, 0.2 , 0.2 , 0.21,
#           0.21, 0.25, 0.25, 0.26, 0.26, 0.32, 0.32, 0.37, 0.37, 0.4 , 0.4 ,
#           0.41, 0.41, 0.45, 0.45, 0.46, 0.46, 0.47, 0.47, 0.53, 0.53, 0.55,
#           0.55, 0.56, 0.56, 0.61, 0.61, 0.69, 0.69, 0.71, 0.71, 1.  ]),
#    'thresholds': [1.9481449127197266,
#     0.9481449127197266,
#     0.9114324450492859,
#     0.9097952842712402,
#     0.8888688087463379,
#     0.8852378726005554,
#     0.8846753835678101,
#     0.8844766020774841,
#     0.8777046799659729,
#     0.8771846294403076,
#     0.8752403259277344,
#     0.8706799745559692,
#     0.8546250462532043,
#     0.8540434837341309,
#     0.8020408153533936,
#     0.7980051040649414,
#     0.7976368069648743,
#     0.7969483137130737,
#     0.794121265411377,
#     0.7866607904434204,
#     0.728582501411438,
#     0.7280181646347046,
#     0.7241008281707764,
#     0.7239280939102173,
#     0.7142009139060974,
#     0.7112079858779907,
#     0.7033612728118896,
#     0.7007791996002197,
#     0.6885221600532532,
#     0.6808049082756042,
#     0.6709012389183044,
#     0.6682149767875671,
#     0.6675899624824524,
#     0.665591299533844,
#     0.6622025966644287,
#     0.6587657332420349,
#     0.655938446521759,
#     0.6552389860153198,
#     0.6523470878601074,
#     0.6511948704719543,
#     0.6395447850227356,
#     0.6392878293991089,
#     0.6317980885505676,
#     0.6261691451072693,
#     0.6073744893074036,
#     0.5845167636871338,
#     0.5825261473655701,
#     0.5783230066299438,
#     0.5480730533599854,
#     0.5334899425506592,
#     0.5312501788139343,
#     0.5181920528411865,
#     0.5171629786491394,
#     0.4872608184814453,
#     0.46472594141960144,
#     0.46420225501060486,
#     0.4513833820819855,
#     0.439253568649292,
#     0.4348732829093933,
#     0.43474724888801575,
#     0.4342431426048279,
#     0.4281863868236542,
#     0.41129112243652344,
#     0.38210606575012207,
#     0.37797337770462036,
#     0.35588011145591736,
#     0.3545667827129364,
#     0.3472338914871216,
#     0.3343864381313324,
#     0.32244816422462463,
#     0.31514474749565125,
#     0.2916359305381775,
#     0.2900457978248596,
#     0.2875465750694275,
#     0.2837088406085968,
#     0.06493230909109116],
#    'tpr': array([0.  , 0.01, 0.09, 0.09, 0.15, 0.15, 0.16, 0.16, 0.18, 0.18, 0.2 ,
#           0.2 , 0.23, 0.23, 0.37, 0.37, 0.38, 0.38, 0.39, 0.39, 0.52, 0.52,
#           0.55, 0.55, 0.58, 0.58, 0.61, 0.61, 0.64, 0.64, 0.65, 0.65, 0.66,
#           0.66, 0.67, 0.67, 0.69, 0.69, 0.7 , 0.7 , 0.72, 0.72, 0.73, 0.73,
#           0.77, 0.77, 0.78, 0.78, 0.82, 0.82, 0.83, 0.83, 0.84, 0.84, 0.85,
#           0.85, 0.87, 0.87, 0.88, 0.88, 0.89, 0.89, 0.91, 0.91, 0.93, 0.93,
#           0.94, 0.94, 0.97, 0.97, 0.98, 0.98, 0.99, 0.99, 1.  , 1.  ])}},
#  'roc_auc_scores': {'Atelectasis': 0.7175,
#   'Cardiomegaly': 0.6975,
#   'Consolidation': 0.6263297872340425,
#   'Edema': 0.7343987823439878,
#   'Enlarged Cardiomediastinum': 0.6807897984368573,
#   'Fracture': 0.5228475228475228,
#   'Lung Lesion': 0.7220744680851064,
#   'Lung Opacity': 0.6454581832733094,
#   'No Finding': 0.7148962148962149,
#   'Pleural Effusion': 0.7291168091168092,
#   'Pleural Other': 0.8793969849246231,
#   'Pneumonia': 0.6348717948717949,
#   'Pneumothorax': 0.550561797752809,
#   'Support Devices': 0.8335}}
def plot_aur_roc_curves():
