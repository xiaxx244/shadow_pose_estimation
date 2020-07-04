from keras.layers import Activation, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Add, Concatenate
from keras.models import Model
from keras.initializers import orthogonal, he_normal
from keras.regularizers import l2
from keras.utils import multi_gpu_model

class MSCNN:
    def __init__(self, num_res_blocks=10, kernel_size=5, nb_channels_in=64):
        self.kernel_size = kernel_size
        self.nb_channels_in = nb_channels_in
        self.num_res_blocks = num_res_blocks

    def residual_block(self, x_input, nb_channels_in, kernel_size=5):
        """
        The residual block take x_input, and the output has the same dimension as x_input
        """
        x_shortcut = x_input

        # first conv + relu
        x = Conv2D(nb_channels_in, (kernel_size, kernel_size),
                   kernel_initializer=orthogonal(), padding='same')(x_input)
        x = Activation('relu')(x)

        # second conv
        x = Conv2D(nb_channels_in, (kernel_size, kernel_size),
                   kernel_initializer=orthogonal(), padding='same')(x)

        # shortcut
        x = Add()([x_shortcut, x])
        return x

    def build_stage_one(self, img_input):
        """
        Build model for stage one in MSCNN
        return: model_stage_1, stage_1_output
        """
        # conv1
        x_stage1 = Conv2D(self.nb_channels_in, (self.kernel_size, self.kernel_size),
                          kernel_initializer=orthogonal(), padding='same')(img_input)

        # chain of residual blocks
        for i in range(self.num_res_blocks):
            x_stage1 = self.residual_block(
                x_stage1, self.nb_channels_in, kernel_size=self.kernel_size)

        # conv2
        x_stage1 = Conv2D(3, (self.kernel_size, self.kernel_size),
                          kernel_initializer=orthogonal(), padding='same')(x_stage1)

        model_stage1 = Model(img_input, x_stage1, name='MSCNN_stage1')
        return model_stage1, x_stage1

    def build_stage_two(self, img_input, prev_input):
        """
        Build model for stage two in MSCNN
        return: model_stage_2, stage_2_output
        """
        # concatenation
        x_stage2 = Concatenate(axis=3)([img_input, prev_input])
        # conv1
        x_stage2 = Conv2D(self.nb_channels_in, (self.kernel_size, self.kernel_size),
                          kernel_initializer=orthogonal(), padding='same')(x_stage2)

        x_shortcut = x_stage2
        # chain of residual blocks
        for i in range(self.num_res_blocks):
            x_stage2 = self.residual_block(
                x_stage2, self.nb_channels_in, kernel_size=self.kernel_size)

        # shortcut
        x_stage2 = Add()([x_stage2, x_shortcut])

        # conv2
        x_stage2 = Conv2D(3, (self.kernel_size, self.kernel_size),
                          kernel_initializer=orthogonal(), padding='same')(x_stage2)

        model_stage2 = Model(img_input, x_stage2, name='MSCNN_stage2')
        return model_stage2, x_stage2

    def build_stage_three(self, img_input, prev_input):
        ######## Third Stage ########
        # concatenation
        x_stage3 = Concatenate(axis=3)([img_input, prev_input])
        # conv1
        x_stage3 = Conv2D(self.nb_channels_in, (self.kernel_size, self.kernel_size),
                          kernel_initializer=orthogonal(), padding='same')(x_stage3)

        # chain of residual blocks
        for i in range(self.num_res_blocks):
            x_stage3 = self.residual_block(
                x_stage3, self.nb_channels_in, kernel_size=self.kernel_size)

        # conv2
        x_stage3 = Conv2D(3, (self.kernel_size, self.kernel_size),
                          kernel_initializer=orthogonal(), padding='same')(x_stage3)

        model_stage3 = Model(img_input, x_stage3, name='MSCNN_stage3')
        return model_stage3, x_stage3

    def build_model(self, input_height, input_width, nChannels):
        inputs = Input(shape=(input_height, input_width, nChannels))
        model_stage_1, output_stage_1 = self.build_stage_one(inputs)
        model_stage_2, output_stage_2 = self.build_stage_two(
            inputs, output_stage_1)
        model_stage_3, output_stage_3 = self.build_stage_three(
            inputs, output_stage_2)

        # Create the overall multi-stage model
        model = Model(inputs=inputs, outputs=[
                      output_stage_1, output_stage_2, output_stage_3], name='MSCNN')

        return multi_gpu_model(model,gpus=2)
