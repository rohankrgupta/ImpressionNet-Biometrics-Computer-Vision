############################# Image Generator for Training task #########################################

def csv_image_generator(inputPath, bs, lb, mode="train", aug=None):
  df = parse_csv(ANNO)
  f = open(inputPath, "r")
  imagePath = "/content/Images/"
  while True:
    images = []
    labels = []
    while len(images) < bs:
      line = f.readline()
      
      if line == "":
        f.seek(0)
        line = f.readline()
        
        if mode == "eval":
          break
      #print(imagePath + line)
      image = cv2.imread(imagePath + line[:-1])
      #print(image)
      image = cv2.resize(image, (224, 224))
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = img_to_array(image)
      images.append(image)
      
      label = df[ATTRIBUTE][line[:-1]]
      labels.append(label)
    
    labels = np.array(labels)
    labels = labels - min(labels)
    labels = np.float32(labels/max(labels))
   
    
    #if data data aug not None
    if aug is not None:
      (images, labels) = next(aug.flow(np.array(images), labels, batch_size=bs))

    

    #print(labels.shape)
    #yeild the batch to the calling function  
    yield(np.array(images), {"model1_output": labels, "model2_output":labels})


