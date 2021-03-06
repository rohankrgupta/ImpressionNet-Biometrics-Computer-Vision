#########################  Coefficient of Determination function  ########################################

def coeff_determination(y_true, y_pred):
  from keras import backend as K
  SS_res =  K.sum(K.square( y_true-y_pred ))
  SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
  return ( 1 - SS_res/(SS_tot + K.epsilon()) )
