import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from llava.model.builder import load_pretrained_model

from PIL import Image
import math
import random

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

# same files as: https://github.com/yuezih/less-is-more/blob/main/CHAIR-eval/data/chair-500.jsonl
image_list = ['COCO_val2014_000000002239.jpg', 'COCO_val2014_000000002302.jpg', 'COCO_val2014_000000003001.jpg', 'COCO_val2014_000000003501.jpg', 'COCO_val2014_000000004312.jpg', 'COCO_val2014_000000004975.jpg', 'COCO_val2014_000000005001.jpg', 'COCO_val2014_000000008211.jpg', 'COCO_val2014_000000012947.jpg', 'COCO_val2014_000000013769.jpg', 'COCO_val2014_000000014056.jpg', 'COCO_val2014_000000015272.jpg', 'COCO_val2014_000000015818.jpg', 'COCO_val2014_000000016502.jpg', 'COCO_val2014_000000019667.jpg', 'COCO_val2014_000000021613.jpg', 'COCO_val2014_000000022423.jpg', 'COCO_val2014_000000022498.jpg', 'COCO_val2014_000000022979.jpg', 'COCO_val2014_000000023709.jpg', 'COCO_val2014_000000027272.jpg', 'COCO_val2014_000000027874.jpg', 'COCO_val2014_000000028156.jpg', 'COCO_val2014_000000028993.jpg', 'COCO_val2014_000000030255.jpg', 'COCO_val2014_000000031521.jpg', 'COCO_val2014_000000036539.jpg', 'COCO_val2014_000000036844.jpg', 'COCO_val2014_000000037017.jpg', 'COCO_val2014_000000037629.jpg', 'COCO_val2014_000000037677.jpg', 'COCO_val2014_000000037734.jpg', 'COCO_val2014_000000040361.jpg', 'COCO_val2014_000000040821.jpg', 'COCO_val2014_000000043530.jpg', 'COCO_val2014_000000045195.jpg', 'COCO_val2014_000000045882.jpg', 'COCO_val2014_000000046633.jpg', 'COCO_val2014_000000051685.jpg', 'COCO_val2014_000000053774.jpg', 'COCO_val2014_000000054277.jpg', 'COCO_val2014_000000054679.jpg', 'COCO_val2014_000000055857.jpg', 'COCO_val2014_000000058754.jpg', 'COCO_val2014_000000060456.jpg', 'COCO_val2014_000000061693.jpg', 'COCO_val2014_000000063328.jpg', 'COCO_val2014_000000063516.jpg', 'COCO_val2014_000000063934.jpg', 'COCO_val2014_000000066297.jpg', 'COCO_val2014_000000066336.jpg', 'COCO_val2014_000000067310.jpg', 'COCO_val2014_000000068852.jpg', 'COCO_val2014_000000071602.jpg', 'COCO_val2014_000000071699.jpg', 'COCO_val2014_000000072281.jpg', 'COCO_val2014_000000072342.jpg', 'COCO_val2014_000000072839.jpg', 'COCO_val2014_000000073139.jpg', 'COCO_val2014_000000073973.jpg', 'COCO_val2014_000000075372.jpg', 'COCO_val2014_000000076292.jpg', 'COCO_val2014_000000077889.jpg', 'COCO_val2014_000000078093.jpg', 'COCO_val2014_000000078495.jpg', 'COCO_val2014_000000079588.jpg', 'COCO_val2014_000000080213.jpg', 'COCO_val2014_000000081956.jpg', 'COCO_val2014_000000082765.jpg', 'COCO_val2014_000000083065.jpg', 'COCO_val2014_000000083113.jpg', 'COCO_val2014_000000083906.jpg', 'COCO_val2014_000000085434.jpg', 'COCO_val2014_000000086001.jpg', 'COCO_val2014_000000086471.jpg', 'COCO_val2014_000000087052.jpg', 'COCO_val2014_000000087419.jpg', 'COCO_val2014_000000087456.jpg', 'COCO_val2014_000000088142.jpg', 'COCO_val2014_000000089603.jpg', 'COCO_val2014_000000092683.jpg', 'COCO_val2014_000000093476.jpg', 'COCO_val2014_000000096251.jpg', 'COCO_val2014_000000096804.jpg', 'COCO_val2014_000000098633.jpg', 'COCO_val2014_000000098853.jpg', 'COCO_val2014_000000098871.jpg', 'COCO_val2014_000000098981.jpg', 'COCO_val2014_000000099996.jpg', 'COCO_val2014_000000100000.jpg', 'COCO_val2014_000000100543.jpg', 'COCO_val2014_000000101785.jpg', 'COCO_val2014_000000104837.jpg', 'COCO_val2014_000000105783.jpg', 'COCO_val2014_000000106048.jpg', 'COCO_val2014_000000106113.jpg', 'COCO_val2014_000000106351.jpg', 'COCO_val2014_000000109216.jpg', 'COCO_val2014_000000110560.jpg', 'COCO_val2014_000000111811.jpg', 'COCO_val2014_000000112066.jpg', 'COCO_val2014_000000114119.jpg', 'COCO_val2014_000000114710.jpg', 'COCO_val2014_000000116202.jpg', 'COCO_val2014_000000117380.jpg', 'COCO_val2014_000000117841.jpg', 'COCO_val2014_000000122588.jpg', 'COCO_val2014_000000124636.jpg', 'COCO_val2014_000000126226.jpg', 'COCO_val2014_000000127496.jpg', 'COCO_val2014_000000128586.jpg', 'COCO_val2014_000000129322.jpg', 'COCO_val2014_000000131661.jpg', 'COCO_val2014_000000132682.jpg', 'COCO_val2014_000000133527.jpg', 'COCO_val2014_000000134133.jpg', 'COCO_val2014_000000135399.jpg', 'COCO_val2014_000000135748.jpg', 'COCO_val2014_000000137810.jpg', 'COCO_val2014_000000137950.jpg', 'COCO_val2014_000000140203.jpg', 'COCO_val2014_000000140292.jpg', 'COCO_val2014_000000141086.jpg', 'COCO_val2014_000000141247.jpg', 'COCO_val2014_000000142360.jpg', 'COCO_val2014_000000142620.jpg', 'COCO_val2014_000000142879.jpg', 'COCO_val2014_000000144305.jpg', 'COCO_val2014_000000146078.jpg', 'COCO_val2014_000000146193.jpg', 'COCO_val2014_000000146738.jpg', 'COCO_val2014_000000151521.jpg', 'COCO_val2014_000000152731.jpg', 'COCO_val2014_000000152751.jpg', 'COCO_val2014_000000153013.jpg', 'COCO_val2014_000000154057.jpg', 'COCO_val2014_000000155270.jpg', 'COCO_val2014_000000155355.jpg', 'COCO_val2014_000000157418.jpg', 'COCO_val2014_000000158548.jpg', 'COCO_val2014_000000158641.jpg', 'COCO_val2014_000000159608.jpg', 'COCO_val2014_000000161622.jpg', 'COCO_val2014_000000162415.jpg', 'COCO_val2014_000000162495.jpg', 'COCO_val2014_000000165056.jpg', 'COCO_val2014_000000165572.jpg', 'COCO_val2014_000000167235.jpg', 'COCO_val2014_000000169166.jpg', 'COCO_val2014_000000169356.jpg', 'COCO_val2014_000000169602.jpg', 'COCO_val2014_000000170311.jpg', 'COCO_val2014_000000170630.jpg', 'COCO_val2014_000000170992.jpg', 'COCO_val2014_000000171622.jpg', 'COCO_val2014_000000172094.jpg', 'COCO_val2014_000000172271.jpg', 'COCO_val2014_000000172595.jpg', 'COCO_val2014_000000173598.jpg', 'COCO_val2014_000000176744.jpg', 'COCO_val2014_000000177064.jpg', 'COCO_val2014_000000178616.jpg', 'COCO_val2014_000000179948.jpg', 'COCO_val2014_000000181466.jpg', 'COCO_val2014_000000182785.jpg', 'COCO_val2014_000000185292.jpg', 'COCO_val2014_000000187533.jpg', 'COCO_val2014_000000187734.jpg', 'COCO_val2014_000000191580.jpg', 'COCO_val2014_000000192440.jpg', 'COCO_val2014_000000193968.jpg', 'COCO_val2014_000000195862.jpg', 'COCO_val2014_000000196074.jpg', 'COCO_val2014_000000196715.jpg', 'COCO_val2014_000000198178.jpg', 'COCO_val2014_000000202810.jpg', 'COCO_val2014_000000203257.jpg', 'COCO_val2014_000000204943.jpg', 'COCO_val2014_000000208377.jpg', 'COCO_val2014_000000208871.jpg', 'COCO_val2014_000000209041.jpg', 'COCO_val2014_000000209048.jpg', 'COCO_val2014_000000212072.jpg', 'COCO_val2014_000000212122.jpg', 'COCO_val2014_000000212403.jpg', 'COCO_val2014_000000215287.jpg', 'COCO_val2014_000000215608.jpg', 'COCO_val2014_000000216428.jpg', 'COCO_val2014_000000217039.jpg', 'COCO_val2014_000000217614.jpg', 'COCO_val2014_000000217846.jpg', 'COCO_val2014_000000218424.jpg', 'COCO_val2014_000000219848.jpg', 'COCO_val2014_000000220171.jpg', 'COCO_val2014_000000220214.jpg', 'COCO_val2014_000000222407.jpg', 'COCO_val2014_000000222991.jpg', 'COCO_val2014_000000223198.jpg', 'COCO_val2014_000000224974.jpg', 'COCO_val2014_000000225518.jpg', 'COCO_val2014_000000226161.jpg', 'COCO_val2014_000000226220.jpg', 'COCO_val2014_000000228011.jpg', 'COCO_val2014_000000228854.jpg', 'COCO_val2014_000000229096.jpg', 'COCO_val2014_000000230240.jpg', 'COCO_val2014_000000231153.jpg', 'COCO_val2014_000000231471.jpg', 'COCO_val2014_000000231732.jpg', 'COCO_val2014_000000233042.jpg', 'COCO_val2014_000000233815.jpg', 'COCO_val2014_000000234169.jpg', 'COCO_val2014_000000234291.jpg', 'COCO_val2014_000000239048.jpg', 'COCO_val2014_000000239980.jpg', 'COCO_val2014_000000244240.jpg', 'COCO_val2014_000000246145.jpg', 'COCO_val2014_000000246252.jpg', 'COCO_val2014_000000247917.jpg', 'COCO_val2014_000000249131.jpg', 'COCO_val2014_000000249838.jpg', 'COCO_val2014_000000252354.jpg', 'COCO_val2014_000000254493.jpg', 'COCO_val2014_000000255209.jpg', 'COCO_val2014_000000256031.jpg', 'COCO_val2014_000000260257.jpg', 'COCO_val2014_000000261757.jpg', 'COCO_val2014_000000263644.jpg', 'COCO_val2014_000000266866.jpg', 'COCO_val2014_000000268229.jpg', 'COCO_val2014_000000268734.jpg', 'COCO_val2014_000000268944.jpg', 'COCO_val2014_000000269682.jpg', 'COCO_val2014_000000270215.jpg', 'COCO_val2014_000000274262.jpg', 'COCO_val2014_000000274455.jpg', 'COCO_val2014_000000275448.jpg', 'COCO_val2014_000000276552.jpg', 'COCO_val2014_000000276804.jpg', 'COCO_val2014_000000279919.jpg', 'COCO_val2014_000000281008.jpg', 'COCO_val2014_000000283037.jpg', 'COCO_val2014_000000283131.jpg', 'COCO_val2014_000000283642.jpg', 'COCO_val2014_000000284064.jpg', 'COCO_val2014_000000285832.jpg', 'COCO_val2014_000000286503.jpg', 'COCO_val2014_000000292082.jpg', 'COCO_val2014_000000292916.jpg', 'COCO_val2014_000000293026.jpg', 'COCO_val2014_000000294605.jpg', 'COCO_val2014_000000295076.jpg', 'COCO_val2014_000000295138.jpg', 'COCO_val2014_000000296492.jpg', 'COCO_val2014_000000296897.jpg', 'COCO_val2014_000000297180.jpg', 'COCO_val2014_000000298562.jpg', 'COCO_val2014_000000300137.jpg', 'COCO_val2014_000000301102.jpg', 'COCO_val2014_000000301691.jpg', 'COCO_val2014_000000302361.jpg', 'COCO_val2014_000000302908.jpg', 'COCO_val2014_000000303250.jpg', 'COCO_val2014_000000306230.jpg', 'COCO_val2014_000000307800.jpg', 'COCO_val2014_000000309232.jpg', 'COCO_val2014_000000311206.jpg', 'COCO_val2014_000000313647.jpg', 'COCO_val2014_000000315128.jpg', 'COCO_val2014_000000315350.jpg', 'COCO_val2014_000000316464.jpg', 'COCO_val2014_000000317015.jpg', 'COCO_val2014_000000317531.jpg', 'COCO_val2014_000000319051.jpg', 'COCO_val2014_000000320743.jpg', 'COCO_val2014_000000322482.jpg', 'COCO_val2014_000000325152.jpg', 'COCO_val2014_000000326781.jpg', 'COCO_val2014_000000327070.jpg', 'COCO_val2014_000000327479.jpg', 'COCO_val2014_000000328777.jpg', 'COCO_val2014_000000330535.jpg', 'COCO_val2014_000000331455.jpg', 'COCO_val2014_000000331785.jpg', 'COCO_val2014_000000333000.jpg', 'COCO_val2014_000000333167.jpg', 'COCO_val2014_000000334509.jpg', 'COCO_val2014_000000335521.jpg', 'COCO_val2014_000000335800.jpg', 'COCO_val2014_000000336892.jpg', 'COCO_val2014_000000337815.jpg', 'COCO_val2014_000000338428.jpg', 'COCO_val2014_000000338562.jpg', 'COCO_val2014_000000340577.jpg', 'COCO_val2014_000000340946.jpg', 'COCO_val2014_000000342593.jpg', 'COCO_val2014_000000343774.jpg', 'COCO_val2014_000000343914.jpg', 'COCO_val2014_000000344773.jpg', 'COCO_val2014_000000345376.jpg', 'COCO_val2014_000000345535.jpg', 'COCO_val2014_000000349437.jpg', 'COCO_val2014_000000350230.jpg', 'COCO_val2014_000000350270.jpg', 'COCO_val2014_000000350341.jpg', 'COCO_val2014_000000350491.jpg', 'COCO_val2014_000000352011.jpg', 'COCO_val2014_000000353398.jpg', 'COCO_val2014_000000353562.jpg', 'COCO_val2014_000000354165.jpg', 'COCO_val2014_000000354744.jpg', 'COCO_val2014_000000356043.jpg', 'COCO_val2014_000000357278.jpg', 'COCO_val2014_000000359833.jpg', 'COCO_val2014_000000360274.jpg', 'COCO_val2014_000000363343.jpg', 'COCO_val2014_000000364256.jpg', 'COCO_val2014_000000364343.jpg', 'COCO_val2014_000000365540.jpg', 'COCO_val2014_000000365623.jpg', 'COCO_val2014_000000365993.jpg', 'COCO_val2014_000000372874.jpg', 'COCO_val2014_000000373284.jpg', 'COCO_val2014_000000375286.jpg', 'COCO_val2014_000000376278.jpg', 'COCO_val2014_000000376793.jpg', 'COCO_val2014_000000376937.jpg', 'COCO_val2014_000000377670.jpg', 'COCO_val2014_000000377999.jpg', 'COCO_val2014_000000378844.jpg', 'COCO_val2014_000000381213.jpg', 'COCO_val2014_000000381551.jpg', 'COCO_val2014_000000383414.jpg', 'COCO_val2014_000000384092.jpg', 'COCO_val2014_000000384111.jpg', 'COCO_val2014_000000384952.jpg', 'COCO_val2014_000000387434.jpg', 'COCO_val2014_000000388157.jpg', 'COCO_val2014_000000388290.jpg', 'COCO_val2014_000000388481.jpg', 'COCO_val2014_000000388564.jpg', 'COCO_val2014_000000388902.jpg', 'COCO_val2014_000000389843.jpg', 'COCO_val2014_000000393254.jpg', 'COCO_val2014_000000394009.jpg', 'COCO_val2014_000000394801.jpg', 'COCO_val2014_000000395557.jpg', 'COCO_val2014_000000396863.jpg', 'COCO_val2014_000000397980.jpg', 'COCO_val2014_000000398818.jpg', 'COCO_val2014_000000399212.jpg', 'COCO_val2014_000000399636.jpg', 'COCO_val2014_000000402519.jpg', 'COCO_val2014_000000405972.jpg', 'COCO_val2014_000000406129.jpg', 'COCO_val2014_000000407067.jpg', 'COCO_val2014_000000412899.jpg', 'COCO_val2014_000000413446.jpg', 'COCO_val2014_000000416326.jpg', 'COCO_val2014_000000416749.jpg', 'COCO_val2014_000000418144.jpg', 'COCO_val2014_000000421923.jpg', 'COCO_val2014_000000422698.jpg', 'COCO_val2014_000000423919.jpg', 'COCO_val2014_000000430273.jpg', 'COCO_val2014_000000431615.jpg', 'COCO_val2014_000000432414.jpg', 'COCO_val2014_000000432519.jpg', 'COCO_val2014_000000433652.jpg', 'COCO_val2014_000000433998.jpg', 'COCO_val2014_000000434509.jpg', 'COCO_val2014_000000434746.jpg', 'COCO_val2014_000000435037.jpg', 'COCO_val2014_000000435358.jpg', 'COCO_val2014_000000436280.jpg', 'COCO_val2014_000000436722.jpg', 'COCO_val2014_000000437363.jpg', 'COCO_val2014_000000440189.jpg', 'COCO_val2014_000000440284.jpg', 'COCO_val2014_000000440299.jpg', 'COCO_val2014_000000441225.jpg', 'COCO_val2014_000000443780.jpg', 'COCO_val2014_000000443844.jpg', 'COCO_val2014_000000445812.jpg', 'COCO_val2014_000000447089.jpg', 'COCO_val2014_000000447611.jpg', 'COCO_val2014_000000449546.jpg', 'COCO_val2014_000000451674.jpg', 'COCO_val2014_000000453557.jpg', 'COCO_val2014_000000453565.jpg', 'COCO_val2014_000000454382.jpg', 'COCO_val2014_000000455448.jpg', 'COCO_val2014_000000456223.jpg', 'COCO_val2014_000000456730.jpg', 'COCO_val2014_000000456908.jpg', 'COCO_val2014_000000456991.jpg', 'COCO_val2014_000000457817.jpg', 'COCO_val2014_000000458275.jpg', 'COCO_val2014_000000458311.jpg', 'COCO_val2014_000000459182.jpg', 'COCO_val2014_000000462904.jpg', 'COCO_val2014_000000464149.jpg', 'COCO_val2014_000000465007.jpg', 'COCO_val2014_000000465275.jpg', 'COCO_val2014_000000465430.jpg', 'COCO_val2014_000000465453.jpg', 'COCO_val2014_000000467580.jpg', 'COCO_val2014_000000468245.jpg', 'COCO_val2014_000000468337.jpg', 'COCO_val2014_000000469192.jpg', 'COCO_val2014_000000472484.jpg', 'COCO_val2014_000000474384.jpg', 'COCO_val2014_000000475229.jpg', 'COCO_val2014_000000476856.jpg', 'COCO_val2014_000000477542.jpg', 'COCO_val2014_000000481654.jpg', 'COCO_val2014_000000484651.jpg', 'COCO_val2014_000000485027.jpg', 'COCO_val2014_000000485613.jpg', 'COCO_val2014_000000488573.jpg', 'COCO_val2014_000000492678.jpg', 'COCO_val2014_000000497117.jpg', 'COCO_val2014_000000500583.jpg', 'COCO_val2014_000000500940.jpg', 'COCO_val2014_000000501790.jpg', 'COCO_val2014_000000502214.jpg', 'COCO_val2014_000000504101.jpg', 'COCO_val2014_000000504304.jpg', 'COCO_val2014_000000505650.jpg', 'COCO_val2014_000000506569.jpg', 'COCO_val2014_000000506945.jpg', 'COCO_val2014_000000507154.jpg', 'COCO_val2014_000000507249.jpg', 'COCO_val2014_000000507427.jpg', 'COCO_val2014_000000508731.jpg', 'COCO_val2014_000000509131.jpg', 'COCO_val2014_000000509699.jpg', 'COCO_val2014_000000513181.jpg', 'COCO_val2014_000000513727.jpg', 'COCO_val2014_000000517612.jpg', 'COCO_val2014_000000517793.jpg', 'COCO_val2014_000000518729.jpg', 'COCO_val2014_000000520147.jpg', 'COCO_val2014_000000520508.jpg', 'COCO_val2014_000000521236.jpg', 'COCO_val2014_000000521540.jpg', 'COCO_val2014_000000522032.jpg', 'COCO_val2014_000000523123.jpg', 'COCO_val2014_000000523696.jpg', 'COCO_val2014_000000525286.jpg', 'COCO_val2014_000000526321.jpg', 'COCO_val2014_000000527846.jpg', 'COCO_val2014_000000528866.jpg', 'COCO_val2014_000000528977.jpg', 'COCO_val2014_000000529102.jpg', 'COCO_val2014_000000530875.jpg', 'COCO_val2014_000000531286.jpg', 'COCO_val2014_000000531771.jpg', 'COCO_val2014_000000531995.jpg', 'COCO_val2014_000000533261.jpg', 'COCO_val2014_000000533511.jpg', 'COCO_val2014_000000534605.jpg', 'COCO_val2014_000000535326.jpg', 'COCO_val2014_000000535536.jpg', 'COCO_val2014_000000535858.jpg', 'COCO_val2014_000000539557.jpg', 'COCO_val2014_000000539717.jpg', 'COCO_val2014_000000541924.jpg', 'COCO_val2014_000000543239.jpg', 'COCO_val2014_000000547341.jpg', 'COCO_val2014_000000548201.jpg', 'COCO_val2014_000000548246.jpg', 'COCO_val2014_000000553141.jpg', 'COCO_val2014_000000553808.jpg', 'COCO_val2014_000000554579.jpg', 'COCO_val2014_000000557107.jpg', 'COCO_val2014_000000558641.jpg', 'COCO_val2014_000000559730.jpg', 'COCO_val2014_000000559838.jpg', 'COCO_val2014_000000561027.jpg', 'COCO_val2014_000000562496.jpg', 'COCO_val2014_000000565954.jpg', 'COCO_val2014_000000566173.jpg', 'COCO_val2014_000000566314.jpg', 'COCO_val2014_000000566824.jpg', 'COCO_val2014_000000567340.jpg', 'COCO_val2014_000000567740.jpg', 'COCO_val2014_000000567812.jpg', 'COCO_val2014_000000570022.jpg', 'COCO_val2014_000000572028.jpg', 'COCO_val2014_000000572178.jpg', 'COCO_val2014_000000572229.jpg', 'COCO_val2014_000000572861.jpg', 'COCO_val2014_000000573206.jpg', 'COCO_val2014_000000573565.jpg', 'COCO_val2014_000000573796.jpg', 'COCO_val2014_000000574706.jpg', 'COCO_val2014_000000574769.jpg', 'COCO_val2014_000000579060.jpg', 'COCO_val2014_000000581655.jpg']


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, additional_input_prompt, 
                 size=500, seed=42):
        # self.questions = questions
        self.image_folder = image_folder

        img_files = os.listdir(image_folder)
        random.seed(seed)
        random.shuffle(img_files)
        self.img_files=img_files[:size]

        # FIXME: make this nicer;
        self.img_files = image_list

        self.questions=self.img_files

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.additional_input_prompt = additional_input_prompt

    def __getitem__(self, index):
        image_file = self.img_files[index]
        qs = self.additional_input_prompt # added by PS
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4, additional_input_prompt='', 
                       size=500, seed=42):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, additional_input_prompt, 
                            size, seed)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    # disable_torch_init()
    # model_path = os.path.expanduser(args.model_path)
    # model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)


    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print('Loading LLaVA...')
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # questions = [json.lZoads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader("", args.image_folder, tokenizer, image_processor, model.config, 
                                        additional_input_prompt=args.additional_input_prompt, 
                                        size=args.num_samples, seed=args.seed)
    
    questions = data_loader.dataset.questions # ugly hack
    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = int(line.split(".jpg")[0][-6:]) # image id
        cur_prompt = args.additional_input_prompt # added by PS

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()

        # # just discard the incomplete sentence
        # if not outputs.endswith('.') and len(outputs.split('.')[:-1]) >0:
        #     outputs = '.'.join(outputs.split('.')[:-1])+'.'


        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/scratch/ssd004/datasets/MSCOCO2014/val2014")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--additional_input_prompt", type=str, default='', 
                help='pass additional input prompt to the question.')
    parser.add_argument("--num_samples", type=int, default=500, help="based on https://github.com/shikiw/OPERA/blob/main/chair_eval.py we take subset of 500., pass -1 to take the whole set.")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    eval_model(args)
