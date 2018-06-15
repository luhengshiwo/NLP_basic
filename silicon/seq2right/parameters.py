class Para:
    embedding_size = 150
    unk = 'uuunnnkkk'
    pad = '<pad>'
    tgt_sos = '\<s>'
    tgt_eos = '\</s>'
    batch_size = 32
    num_units = 100
    max_gradient_norm = 5
    learning_rate = 0.001
    n_epochs = 20
    n_outputs = 1
    train_keep_prob = 0.5
    # train_num = 7478
    train_num = 512   
    dev_num = 256
    test_num = 256
    threshold = 1.0
    l2_rate = 0.0001
    beam_width = 10


