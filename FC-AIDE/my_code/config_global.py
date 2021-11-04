import argparse

parser_global = argparse.ArgumentParser()
parser_global.add_argument('--input_size', default=(320, 320, 3))

parser_global.add_argument('--data_dir', default='./my_code/data/')

parser_global.add_argument('--model_making_parser', default=[[64, 1, 1, 0], *[[64, 2, i+2, 1] for i in range(21)]])
parser_global.add_argument('--model_making_aa_out_channel', default=128)
parser_global.add_argument('--model_making_parser_other', default=[[128, ], [128, ]])
parser_global.add_argument('--model_making_end_out_channel', default=2)
parser_global.add_argument('--model_making_str_template_1', default="'Conv2dFilter(' + str(parse[0]) + ', filter_mode=\"' + str(i) + str(parse[1]) + '\")(value[i])'")
parser_global.add_argument('--model_making_str_template_2', default="'Activation(\"relu\")(Average()(value))'")
parser_global.add_argument('--model_making_str_template_3', default="'Activation(\"relu\")(Average()([Conv2D(' + str(parse[0]) + ',(1,1), kernel_initializer=\"he_uniform\")(Activation(\"relu\")(Conv2D(' + str(parse[0]) + ',(1,1), kernel_initializer=\"he_uniform\")(avg))), avg]))'")
parser_global.add_argument('--model_making_str_template_4', default="'Activation(\"relu\")(value[i])'")
parser_global.add_argument('--model_loss', default='fine_tuning_loss')
parser_global.add_argument('--model_optimizer', default='adam')

parser_global = parser_global.parse_args()

