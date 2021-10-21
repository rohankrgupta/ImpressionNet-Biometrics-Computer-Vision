
#######################  Multi-view Model Class ###########################################

class ImpressionNet:
  @staticmethod
  def build_model1(inputs):
    # Base Model
    base_model1 = EfficientNetB0(
    weights = 'imagenet',
    input_shape = (224, 224, 3),
    include_top = False,
    drop_connect_rate=0.3,
    #pooling = max,
    input_tensor = inputs
    )
    # Set train to false
    base_model1.trainable = True
    
    #Get the last Layer
    last_layer1 = base_model1.get_layer('block7a_se_squeeze')
    last_output1 = last_layer1.output
    
    # Flatten the output layer to 1 dimension
    x1 = layers.Flatten()(last_output1)
    x1 = layers.Dense(1024, activation='relu', name = 'denselayer1')(x1)
    x1 = layers.Dense(1, activation='linear', name = "model1_output")(x1)
    
    return x1

  @staticmethod
  def build_model2(inputs):
    # Base Model
    base_model2 = EfficientNetB0(
    weights = 'imagenet',
    input_shape = (224, 224, 3),
    include_top = False,
    drop_connect_rate=0.3,
    #pooling = max,
    input_tensor = inputs
    )

    # Set train to false
    base_model2.trainable = True
    
    for layer in base_model2.layers:
      layer._name = 'new' + str(layer._name)

    #Get the last Layer
    last_layer2 = base_model2.get_layer('newblock7a_se_squeeze')
    last_output2 = last_layer2.output
    
    # Flatten the output layer to 1 dimension
    x2 = layers.Flatten()(last_output2)
    x2 = layers.Dense(1024, activation='relu', name = 'denselayer2')(x2)
    x2 = layers.Dense(1, activation='linear', name = "model2_output")(x2)
   
    return x2  

  @staticmethod
  def build():
    
    inputShape = (224, 224, 3)
    inputs = tf.keras.Input(shape = inputShape)
    
    model1 = ImpressionNet.build_model1(inputs)
    model2 = ImpressionNet.build_model2(inputs)

    model = Model(
        inputs = inputs,
        outputs = [model1, model2],
        name='impressionnet'
    )

    #return the constructed arhitecture
    return model
    
