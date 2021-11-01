# -*- coding: utf-8 -*-
import random
import math
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6, 5) # 单位是inches
plt.rcParams['boxplot.flierprops.markersize'] = 3 # 默认为6


x1 = np.linspace(20, 100, 5)
x2 = range(100)

GA_times_20 = [1.3969528675079346, 1.458542823791504, 1.457716941833496, 1.4201502799987793, 1.432178020477295, 1.3461995124816895, 1.361063003540039, 1.3963816165924072, 1.5679128170013428, 1.62416672706604, 1.5182275772094727, 1.4822299480438232, 1.3535590171813965, 1.355536937713623, 1.3838317394256592, 1.3851885795593262, 1.3472967147827148, 1.3669238090515137, 1.3886802196502686, 1.3960981369018555, 1.330843210220337, 1.3421382904052734, 1.4029171466827393, 1.4681751728057861, 1.4086980819702148, 1.3523674011230469, 1.3684391975402832, 1.3631725311279297, 1.35809326171875, 1.3588850498199463, 1.3864481449127197, 1.4253995418548584, 1.36993408203125, 1.4043588638305664, 1.3862087726593018, 1.365952968597412, 1.3704509735107422, 1.3812615871429443, 1.4331011772155762, 1.4126083850860596, 1.3931093215942383, 1.3614416122436523, 1.3672966957092285, 1.3966543674468994, 1.3719041347503662, 1.4157581329345703, 1.3184268474578857, 1.3701300621032715, 1.3956594467163086, 1.3419404029846191, 1.395007610321045, 1.3349647521972656, 1.39738130569458, 1.3633337020874023, 1.3925738334655762, 1.3980700969696045, 1.3999640941619873, 1.3402297496795654, 1.344050407409668, 1.3256831169128418, 1.3914766311645508, 1.3426144123077393, 1.4196171760559082, 1.380864143371582, 1.3605904579162598, 1.415595531463623, 1.3653030395507812, 1.3170254230499268, 1.3777830600738525, 1.6367762088775635, 1.4227604866027832, 1.415926218032837, 1.3635869026184082, 1.3871610164642334, 1.344482421875, 1.3638300895690918, 1.3675038814544678, 1.4005300998687744, 1.4127206802368164, 1.3575148582458496, 1.3968119621276855, 1.3531389236450195, 1.3592123985290527, 1.5564007759094238, 1.682434320449829, 1.5514225959777832, 1.4001798629760742, 1.757129192352295, 1.6070120334625244, 1.5335874557495117, 1.5422022342681885, 1.8549048900604248, 1.43178129196167, 1.3773303031921387, 1.768874168395996, 1.3819670677185059, 1.6573553085327148, 1.406827688217163, 1.366990566253662, 1.3775675296783447]
GA_times_40 = [3.1655232906341553, 2.845085382461548, 2.788099527359009, 2.760359764099121, 2.8679680824279785, 2.8030643463134766, 2.778263807296753, 2.882816791534424, 3.184305429458618, 3.006838321685791, 2.836040496826172, 2.862942934036255, 3.1369688510894775, 2.908241033554077, 2.9439618587493896, 2.85268235206604, 3.242401361465454, 3.0714850425720215, 3.081979274749756, 3.012505531311035, 2.871027708053589, 2.794703960418701, 2.792156934738159, 2.8206303119659424, 2.898026466369629, 2.841301679611206, 2.8576269149780273, 2.9068851470947266, 3.052664041519165, 3.0321381092071533, 3.1279735565185547, 3.024641752243042, 3.0862910747528076, 2.8189001083374023, 2.996835708618164, 2.719572067260742, 2.7385034561157227, 2.758535385131836, 2.883279323577881, 3.051193952560425, 3.166675090789795, 2.9066824913024902, 2.8741772174835205, 2.837390422821045, 2.7761030197143555, 2.843324899673462, 2.8394923210144043, 2.782468318939209, 2.798567056655884, 2.9118051528930664, 2.807598114013672, 2.7627112865448, 2.869168281555176, 2.8918135166168213, 2.7600350379943848, 2.7896580696105957, 2.9210128784179688, 2.868603467941284, 2.740969181060791, 2.8726048469543457, 2.924377918243408, 2.794433355331421, 2.7261695861816406, 3.00795578956604, 2.897986888885498, 2.7568488121032715, 2.779543161392212, 2.8531155586242676, 2.8612077236175537, 2.7779417037963867, 2.892322301864624, 2.880725145339966, 2.798882246017456, 2.7355520725250244, 2.773296356201172, 2.7208166122436523, 2.749769687652588, 2.8201189041137695, 3.1077189445495605, 3.2874488830566406, 3.0667099952697754, 2.8713760375976562, 3.0044445991516113, 2.8435840606689453, 2.8504445552825928, 2.7814745903015137, 2.732909679412842, 2.7349154949188232, 2.9652583599090576, 3.1922714710235596, 2.8960115909576416, 2.893324613571167, 2.7936151027679443, 2.82266902923584, 2.907362461090088, 2.883604049682617, 2.8397057056427, 2.8539199829101562, 2.9408907890319824, 3.003986120223999]
GA_times_60 = [4.414588928222656, 4.385822296142578, 4.429111003875732, 4.541815519332886, 4.387495994567871, 4.620589733123779, 4.576596736907959, 4.684104681015015, 4.634357690811157, 4.610673904418945, 4.6419148445129395, 4.534411668777466, 4.613491773605347, 4.739660739898682, 4.547332525253296, 5.024709463119507, 4.767440319061279, 4.730140924453735, 4.595045804977417, 5.117020130157471, 4.828668117523193, 4.582614898681641, 4.538147211074829, 4.509271860122681, 4.3852550983428955, 4.590160369873047, 4.621567249298096, 4.384885311126709, 4.651715040206909, 4.525884389877319, 4.545423746109009, 4.674437046051025, 4.589300870895386, 4.448667287826538, 4.702617406845093, 4.602609634399414, 4.567741394042969, 4.49273943901062, 4.462925672531128, 4.388923645019531, 4.632116079330444, 5.075758695602417, 5.120893478393555, 4.681907892227173, 4.700189590454102, 4.316062927246094, 4.400034427642822, 4.96931004524231, 4.969778299331665, 4.587041616439819, 4.403660774230957, 4.435741186141968, 4.433425188064575, 4.44493842124939, 4.555176258087158, 4.607132196426392, 4.383233308792114, 4.497296094894409, 4.512896776199341, 4.46514892578125, 4.440439939498901, 4.6191229820251465, 4.868452787399292, 4.491931676864624, 4.605057001113892, 4.5576698780059814, 4.5209877490997314, 4.541243076324463, 4.540060997009277, 4.486664533615112, 4.720243215560913, 4.723571538925171, 4.628103971481323, 4.7895119190216064, 4.624088764190674, 4.451760768890381, 4.5166544914245605, 4.725559949874878, 4.5535430908203125, 4.514462232589722, 4.480808258056641, 4.509180068969727, 4.463709354400635, 4.437328100204468, 4.6112329959869385, 4.605180978775024, 4.560582160949707, 4.7401604652404785, 4.513076066970825, 4.614029884338379, 4.7565131187438965, 5.395578622817993, 4.812250137329102, 4.508002758026123, 4.4607508182525635, 4.7790069580078125, 4.696727275848389, 4.7311601638793945, 4.496100425720215, 4.542770147323608]
GA_times_80 = [6.353890895843506, 6.386888027191162, 6.4079060554504395, 6.349733352661133, 6.340715646743774, 6.45493221282959, 6.53082275390625, 6.362273216247559, 6.376564979553223, 7.4059014320373535, 6.929109334945679, 6.481393098831177, 6.606434345245361, 6.4499735832214355, 6.547494411468506, 6.440497398376465, 6.237924814224243, 6.286036729812622, 6.44891881942749, 6.996861934661865, 6.325911045074463, 6.49645733833313, 6.772978067398071, 7.112720012664795, 6.504064321517944, 6.352475881576538, 6.493544340133667, 6.290245532989502, 6.191083669662476, 6.4861040115356445, 6.826552867889404, 7.080544710159302, 6.580901861190796, 6.380096673965454, 6.506701707839966, 6.323277473449707, 6.232227802276611, 6.504865646362305, 6.667555332183838, 6.390430212020874, 6.237545728683472, 6.164659261703491, 6.250784397125244, 6.347039222717285, 6.099126815795898, 6.482674837112427, 6.170950174331665, 6.189401626586914, 6.3178629875183105, 6.116309881210327, 6.25515341758728, 6.387727499008179, 6.199815034866333, 6.2659595012664795, 6.279278755187988, 6.246417999267578, 6.224975347518921, 6.146355628967285, 6.160134315490723, 6.407130002975464, 6.518597602844238, 6.096336841583252, 6.064232349395752, 6.2178239822387695, 6.822681188583374, 6.314732789993286, 6.167064189910889, 6.43988823890686, 6.271878242492676, 6.409027338027954, 6.336028337478638, 6.364248037338257, 6.423274993896484, 6.134094476699829, 6.326162815093994, 6.685912132263184, 6.389575719833374, 6.471053600311279, 6.629248857498169, 6.474488019943237, 6.698068857192993, 6.443499803543091, 6.443275213241577, 6.1649229526519775, 6.5407867431640625, 6.284699201583862, 6.441851377487183, 6.497652769088745, 6.250158309936523, 6.665235757827759, 7.119364500045776, 6.493541240692139, 6.456512689590454, 6.575841903686523, 6.500427484512329, 6.722508907318115, 6.93083119392395, 6.380971908569336, 6.522057771682739, 6.582670211791992]
GA_times_100 = [9.161435842514038, 8.775979995727539, 8.850865125656128, 8.801481246948242, 8.81997036933899, 8.938396215438843, 8.949527502059937, 8.877733945846558, 8.93574047088623, 8.808792114257812, 8.691312074661255, 9.341702222824097, 8.894371509552002, 8.756537914276123, 8.871409177780151, 8.976290464401245, 8.800493955612183, 8.91395378112793, 8.894551992416382, 8.96211552619934, 8.865902423858643, 8.908938646316528, 8.941956758499146, 8.874435901641846, 8.82549524307251, 8.874871015548706, 8.858282327651978, 8.786914825439453, 8.776495456695557, 8.817050695419312, 8.819798946380615, 8.7830491065979, 8.832809686660767, 8.77547001838684, 8.819539785385132, 8.930274248123169, 8.922253131866455, 8.846099138259888, 8.88956904411316, 8.819826126098633, 8.806240558624268, 8.870620727539062, 8.811880588531494, 8.844067573547363, 8.999666690826416, 8.826879978179932, 8.778223752975464, 8.790818691253662, 8.926499605178833, 9.129402875900269, 8.92163610458374, 8.756362199783325, 8.82421326637268, 9.189913034439087, 8.726485252380371, 8.801698923110962, 8.77361273765564, 8.768903970718384, 8.970714569091797, 8.788306713104248, 8.855000972747803, 9.310903072357178, 8.940700769424438, 8.821049928665161, 8.720432758331299, 9.294857501983643, 9.03958511352539, 8.86549186706543, 8.852288484573364, 8.938558340072632, 8.916378736495972, 8.930056095123291, 9.030064821243286, 8.69605541229248, 8.749974012374878, 8.797542095184326, 9.32921290397644, 8.90569543838501, 8.923336029052734, 8.797103881835938, 8.801076412200928, 8.926689147949219, 8.842698574066162, 8.967448711395264, 9.439245700836182, 9.713357925415039, 9.06736135482788, 9.030680418014526, 8.926374673843384, 9.507763147354126, 8.867839336395264, 8.734116077423096, 9.180147886276245, 8.934938430786133, 8.79659104347229, 9.121848821640015, 9.070590496063232, 9.08121132850647, 8.752484798431396, 8.955323219299316]
GA_times_20 = np.array(GA_times_20) * 100
GA_times_40 = np.array(GA_times_40) * 100
GA_times_60 = np.array(GA_times_60) * 100
GA_times_80 = np.array(GA_times_80) * 100
GA_times_100 = np.array(GA_times_100) * 100


GA_times_50 = [3.8838164806365967, 3.900123357772827, 3.8593392372131348, 3.7213950157165527, 3.7989017963409424, 3.818812608718872, 3.770413398742676, 3.7801074981689453, 3.813699722290039, 3.785482883453369, 3.7931923866271973, 3.7755703926086426, 3.770000457763672, 3.7651889324188232, 3.7787773609161377, 3.809612512588501, 3.9531309604644775, 3.881732702255249, 3.887235403060913, 3.932215452194214, 3.7209386825561523, 3.76845645904541, 3.7728538513183594, 3.9630489349365234, 3.756800651550293, 3.8427348136901855, 3.8488619327545166, 3.7912330627441406, 3.9556148052215576, 3.9143383502960205, 4.188682556152344, 3.877180576324463, 3.743009567260742, 3.9108920097351074, 3.9301788806915283, 3.7728986740112305, 3.9259536266326904, 3.798004388809204, 3.8543288707733154, 3.8808298110961914, 3.7396368980407715, 3.823650598526001, 3.8684725761413574, 3.743481159210205, 3.942304849624634, 3.798401355743408, 3.793793201446533, 3.8643360137939453, 3.7451608180999756, 3.8802506923675537, 3.8078179359436035, 3.726675510406494, 3.974634885787964, 3.8309381008148193, 3.8127918243408203, 3.902754783630371, 3.7705442905426025, 3.87119197845459, 3.8556385040283203, 3.8171777725219727, 3.821018934249878, 3.7298574447631836, 3.7848196029663086, 4.147101402282715, 4.354912996292114, 3.988614082336426, 3.80431866645813, 3.777031898498535, 3.982778549194336, 3.7448513507843018, 3.8427770137786865, 3.897385358810425, 3.7413268089294434, 3.882061004638672, 3.8143398761749268, 3.721909523010254, 3.8596091270446777, 3.7586545944213867, 3.7981560230255127, 3.876232147216797, 3.711824893951416, 3.8760054111480713, 3.8308777809143066, 3.7517693042755127, 3.9608638286590576, 3.7678725719451904, 3.833021402359009, 3.85709547996521, 3.773022174835205, 3.8620541095733643, 3.8436179161071777, 3.823981761932373, 3.8844826221466064, 3.7823445796966553, 3.7681527137756348, 3.8395895957946777, 3.770829916000366, 3.814636707305908, 3.8001716136932373, 3.7212514877319336]
GA_times_50 = np.array(GA_times_50) * 100


# 画箱线图
# 设置图形的显示风格
# plt.style.use('ggplot')
color = dict(boxes='DarkRed', whiskers='DarkOrange', medians='Black', caps='Gray')

data = {
    '20': GA_times_20,
    '40': GA_times_40,
    '60': GA_times_60,
    "80": GA_times_80,
    "100": GA_times_100
}
df = pd.DataFrame(data)
tmp = df.pop("100")
df.insert(4, "100", tmp)

df.plot.box(title="example show", color=color, showfliers=False)


plt.grid(linestyle = "--")      #设置背景网格线为虚线
ax = plt.gca()
ax.spines['top'].set_visible(False)  #去掉上边框
ax.spines['right'].set_visible(False) #去掉右边框
# plt.grid(linestyle="--", alpha=0.3)

plt.xlabel('Number of Elastic Scaling Instances', fontsize=15, fontweight='bold')
plt.ylabel('Time Consumption(ms)', fontsize=15, fontweight='bold')
plt.title('Stability of DMESM-GA', fontsize=15, fontweight='bold')
plt.xlim(0, 6)
# plt.ylim(21, 38)
plt.tick_params(labelsize=15)
# plt.legend()
# plt.savefig('./Stability of DMESM-GA_1.png', bbox_inches='tight')
plt.show()


# # 画GA_times_50的折线图
# plt.figure(figsize=(6, 4))
# plt.grid(linestyle = "--")      #设置背景网格线为虚线
# ax = plt.gca()
# ax.spines['top'].set_visible(False)  #去掉上边框
# ax.spines['right'].set_visible(False) #去掉右边框
#
# plt.plot(x2, GA_times_50, label='DMESM-SA', linewidth=1, color='DarkRed')
# plt.xlabel('Serial Number', fontsize=15, fontweight='bold')
# plt.ylabel('Time Consumption(ms)', fontsize=15, fontweight='bold')
# plt.title('Time Consumption of DMESM-GA', fontsize=15, fontweight='bold')
# plt.xlim(0, 100)
# plt.ylim(350, 450)
# plt.tick_params(labelsize=15)
# plt.savefig('./Time Consumption of DMESM-GA_1.png', bbox_inches='tight')
# plt.show()