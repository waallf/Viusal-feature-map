    def _concact_features(self, conv_output):
        """
        对特征图进行reshape拼接
        :param conv_output:输入多通道的特征图
        :return:
        """
        num_or_size_splits = conv_output.get_shape().as_list()[-1]
        each_convs = tf.split(conv_output, num_or_size_splits=num_or_size_splits, axis=3)
        concact_size = int(math.sqrt(num_or_size_splits) / 1)
        all_concact = None
        for i in range(concact_size):
            row_concact = each_convs[i * concact_size]
            for j in range(concact_size - 1):
                row_concact = tf.concat([row_concact, each_convs[i * concact_size + j + 1]], 1)
            if i == 0:
                all_concact = row_concact
            else:
                all_concact = tf.concat([all_concact, row_concact], 2)

        return all_concact

    def build_summaries(self,end_points):
        with tf.name_scope('CNN_outputs'):
            tf.summary.image("1_image",self.data["ims"][:,:,:,:],2)
            tf.summary.image("2roi_img",end_points["roi_img"][:,:,:,:],2)
            tf.summary.image('aconv1_1', self._concact_features(end_points["conv1_1"][:, :, :,0:64]), 2)
            tf.summary.image('bconv1_2', self._concact_features(end_points['conv1_2'][:, :, :,0:64]), 2)
            tf.summary.image('cpool1', self._concact_features(end_points['pool1'][:, :, :,0:64]), 2)
            tf.summary.image('dconv2_1', self._concact_features(end_points['conv2_1'][:, :, :,0:128]), 2)
            tf.summary.image('econv2_2', self._concact_features(end_points['conv2_2'][:, :, :,0:128]), 2)
            tf.summary.image('fconv3_1', self._concact_features(end_points['conv3_1'][:, :, :,0:128]), 2)
            tf.summary.image('gconv3_2', self._concact_features(end_points['conv3_2'][:, :, :,0:256]), 2)
            tf.summary.image('hconv3_3', self._concact_features(end_points['conv3_3'][:, :, :,0:256]), 2)
            tf.summary.image('ipool3', self._concact_features(end_points['pool3'][:, :, :,0:256]), 2)
