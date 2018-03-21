 model = C.layers.Sequential([C.layers.Embedding(10), 
    C.layers.Recurrence(C.layers.GRU(10)),C.sequence.last, 
    C.layers.Dense(1, activation=C.sigmoid)])


c = C.Axis.new_unique_dynamic_axis('c')
inp_ph = C.sequence.input_variable(5,sequence_axis=c)
gt_ph = C.sequence.input_variable(1)

cls_scores = model(inp_ph)
gt = C.sequence.last(gt_ph)
loss = C.binary_cross_entropy(cls_scores, gt)

trainer = C.Trainer(cls_scores,(loss,None),C.sgd(cls_scores.parameters, 0.001))

inp=np.random.rand(4,5,5)
labels = np.array([[[1]],[[0]],[[1]],[[1]]])
trainer.train_minibatch({inp_ph:inp, gt_ph:labels})