############################## Combined Loss Function includes MSE and LMV  #############################


def multiview_loss(layer1, layer2):
  
  ########### Multi-View Loss #############
  weight1 = layer1.get_weights()[0]
  weight2 = layer2.get_weights()[0]

  temp1 = tf.norm(weight1)
  temp2 = tf.norm(weight2)

  feat_norm1 = weight1/temp1
  feat_norm2 = weight2/temp2

  x = feat_norm1*feat_norm2
  x = keras.backend.sum(x, 1)
  lmv = tf.keras.metrics.Mean()(abs(x))
  ##########################################
  
  ms = tf.keras.losses.MeanSquaredError()

  def loss(y_true, y_pred):
    return lmv/2 + ms(y_true, y_pred)
  
  return loss



