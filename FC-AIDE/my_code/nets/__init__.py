from tensorflow.keras import Model, Input, backend as K, optimizers
from tensorflow.keras.layers import Average, Activation, Conv2D

from my_code.config_global import parser_global
from my_code.nets.layers import Conv2dFilter


def making_residual(opt, avg, parse):
    return eval(eval(opt.model_making_str_template_3))


def helper_for_making(opt, value, parse):
    for i in range(len(value)):
        if parse[3] > 0:
            value[i] = eval(eval(opt.model_making_str_template_4))
        value[i] = eval(eval(opt.model_making_str_template_1))
    avg = eval(eval(opt.model_making_str_template_2))
    avg_1by1 = making_residual(opt, avg, parse)
    return value, avg_1by1


def get_loss(opt):
    assert opt.model_loss == 'fine_tuning_loss', "The network loses the soul of the loss function."
    return lambda y_true, y_pred : K.mean(K.square(y_true[:, :, :, 1]-(y_pred[:, :, :, 0]*y_true[:, :, :, 1]+y_pred[:, :, :, 1])) + 2*y_pred[:, :, :, 0]*K.square(y_true[:, :, :, 2]) - K.square(y_true[:, :, :, 2]))


def make_model(opt):
    the_input = Input(opt.input_size)
    x_value = [the_input, the_input, the_input]
    avg_1by1_s = []
    for stage_parse in opt.model_making_parser:
        x_value, avg_1by1 = helper_for_making(opt, x_value, stage_parse)
        avg_1by1_s.append(avg_1by1)
    avg_avg_1by1 = Average()(avg_1by1_s)
    x = Conv2D(opt.model_making_aa_out_channel, (1, 1), kernel_initializer='he_uniform')(avg_avg_1by1)
    x_s = [Activation('relu')(x), ]
    for stage_parse in opt.model_making_parser_other:
        x_s.append(making_residual(opt, x_s[0], stage_parse))
    x = Average()(x_s)
    the_output = Conv2D(opt.model_making_end_out_channel, (1, 1), kernel_initializer='he_uniform')(x)

    net = Model([the_input, ], [the_output, ])
    loss = get_loss(opt)
    opti = optimizers.get(opt.model_optimizer)
    net.compile(loss=loss, optimizer=opti)

    return net
