import sys
import models.baechiTest_dummyModels as dm

## To add a new model just scroll down to see a template "elif model_name =....."

def get_model(model_name, repetable, with_gpu_split=False):
    # repetable =1 -> some models can be initialized with constant parameters
    # with_gpu_split=True -> used to get transfomer models already split across gpus

    #### Some settings to ensure values don't go Nan while training
    fct = 6
    lr = 0.0001
    inpt_factor = 0.0000000001
    ##########################

    model_info = {} # for model specific info required during training

    ## Some model dependent setting adjustments
    if model_name == "inception_dummy" and repetable == 0:
        repetable = 1
        print("NOTE: inception_dummy is repetable. Repetable set to 1")
    if model_name == "inception_v3" and repetable == 1:
        repetable = 0
        print("NOTE: inception_v3 is NOT repetable. Repetable set to 0")

    ###################################################################################################
    ## importing appropriate models and info
    
    ## 1. GNMT
    if model_name == "gnmt":
        import models.gnmt as gnmt 

        vocab_size = gnmt.vocab_size  
        max_sequence_length = gnmt.max_sequence_length
        model = gnmt.GNMT(vocab_size, batch_first=True)
        lr = 0.0000000000001
        inp_size_single =(1,1) #gnmt needs no inp_size or opt_size
        opt_size = 0
        model_info['vocab_size'] = vocab_size
        model_info['max_sequence_length'] = max_sequence_length
        model_info['min_sequence_length'] = gnmt.min_sequence_length

    ## 2. Transfomer (Change transformer_type to get specific parts of a transformer instead of the whole)
    elif model_name == "transformer":

        transformer_type = "Transformer"
        #transformer_type = "Decoder"
        unboxed = False # unboxed = each head is pytorch module

        ####### import transformer
        if repetable:
            import models.transformer_repetable as transformer
        else:
            if with_gpu_split:
                print("Expert placement for transformer used")
                import models.transformer_expert as transformer
            else:
                import models.transformer as transformer

        ######## transformer setting 
        ## "Attention is all you need" - paper settings
        n_src_vocab = transformer.vocab_size #30k
        n_trg_vocab = transformer.vocab_size
        src_pad_idx = 0
        trg_pad_idx = 0
        d_word_vec  = 512
        d_model     = 512
        d_inner     = 2048 #(dff in paper)
        n_layers    = 6
        n_head      = 8 #(h in paper)
        d_k         = 64
        d_v         = 64
        ######
        seq_len     = transformer.seq_length #50
        trg_emb_prj_weight_sharing=False
        emb_src_trg_weight_sharing=False

        model_info['vocab_size'] =  transformer.vocab_size
        model_info['trg_emb_prj_weight_sharing'] = trg_emb_prj_weight_sharing
        model_info['emb_src_trg_weight_sharing'] = emb_src_trg_weight_sharing
        
        ##### Chooose specific part of the transformer if needed
        if transformer_type == "ScaledDotProductAttention":
            #***** temperature = 1 (dummy)
            model = transformer.ScaledDotProductAttention(1)
            #**** size of q, k, v each is batch x n_head x seq_len x dv = batch x 8 x 50 x 64
            inp_size = [(8,50,64),(8,50,64),(8,50,64), (8,50,50)] # using mask (size same as attention = seq_len*seq_len)
            opt_size = (8,50,64)
        elif transformer_type == "MultiHeadAttention":
            model = transformer.MultiHeadAttention(8, 512, 64, 64, dropout=0.1).to(0) #64 = 512/8
            #**** size of q, k, v each is batch x seq_len x (dv*n_head) = batch x 50 x (8*64)
            inp_size = [(50,512),(50,512),(50,512), (50,50)] #with mask
            opt_size = (50,512)
        elif transformer_type == "EncoderLayer":
            model = transformer.EncoderLayer(512, 2048, 8, 64, 64, dropout=0.1, unboxed=False)
            inp_size = [(50,512), (50,50)] # with mask
            opt_size = (50,512)
        elif transformer_type == "DecoderLayer":
            model = transformer.DecoderLayer(512, 2048, 8, 64, 64, dropout=0.1, unboxed=False)
            inp_size = [(50,512),(50,512), (50,50), (50,50)] # with enc and dec mask 
            opt_size = (50,512)
        elif transformer_type == "Encoder":
            #**** n_src_vocab=30k , d_word_vec=d_model, n_layers=6, n_head=8, d_k= (512/8)=64 = d_v, d_model=512, d_inner=2048, pad_idx = 0 (some integer)
            model = transformer.Encoder(30000, 512, 6, 8, 64, 64, 512, 2048, 0, dropout=0.1, n_position=200, scale_emb=False, unboxed=unboxed)
            inp_size = [("int",(50,)), (50,50)] # (seq_len,) for input, and (seq_len,seq_len) for mask size 
            opt_size = (50,512)
        elif transformer_type == "Decoder":
            #**** n_trg_vocab=30k, d_word_vec=d_model, n_layers=6, n_head=8, d_k=64, d_v=64, d_model=512, d_inner=2048, pad_id=0
            model = transformer.Decoder(n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, 0, n_position=200, dropout=0.1, scale_emb=False, unboxed=unboxed)
            inp_size = [("int",(50,)), (50,50), (50,512), (50,50) ] # trg_seq, trg_mask, enc_output, src_mask
            opt_size = (50,512)
        elif transformer_type == "Transformer":
            model = transformer.Transformer(n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
                        d_word_vec, d_model, d_inner,
                        n_layers, n_head, d_k, d_v, dropout=0.1, n_position=200,
                        trg_emb_prj_weight_sharing=trg_emb_prj_weight_sharing, emb_src_trg_weight_sharing=emb_src_trg_weight_sharing,
                        scale_emb_or_prj='prj', unboxed=unboxed)    
            inp_size = [("int",(seq_len,)), ("int",(seq_len,))] # (seq_len,) for src and trg sequences
            opt_size = (seq_len,n_trg_vocab)
            
        inp_size_single =   inp_size     

    ## 3. Inception v3
    elif model_name == "inception_v3":
        if repetable: # this doen't work #TODO: make models.inception_modified_predictable_v2 predictable
            import models.inception_modified_predictable_v2 as inception_modified_predictable
            model = inception_modified_predictable.inception_v3(pretrained=True)
        else:
            import models.inception_modified_v2 as inception_model
            model = inception_model.inception_v3(pretrained=True)

        inp_size_single = (3, 299, 299)
        opt_size = 1000
        lr = 0.001

    ## 4. Other test inception_v3 like models  
    elif model_name == "inception_dummy":
        model = dm.inception3(1000, repetable)
        inp_size_single = (3, 299, 299)
        opt_size = 1000
        inpt_factor = 0.01
        lr = 800 # to be used if only upto 5c layers are used
        #inpt_factor = 1000
        #lr = 0.001

    elif model_name == "InceptionE":
        model = dm.inceptionE3(1280, repetable)
        inp_size_single = (1280, 17, 17)
        opt_size = 2048
        inpt_factor = 0.001 # if run_type == "training"
        
    elif model_name == "InceptionE2":
        model = dm.inceptionE2(1280, repetable)
        inp_size_single = (1280, 17, 17)
        opt_size = 1280
        lr = 0.001
        
    elif model_name == "ShortInceptionE":
        model = dm.shortInceptionE(512, repetable)
        inp_size_single = (512, 32, 32)
        opt_size = 512
        
    elif model_name == "TallParallelModel":
        factor = fct
        inp_size_single = (1, 512*factor)
        model = dm.tallParallelModel(factor, repetable)
        opt_size = 1000
        
    elif model_name == "ParallelTwoLayer":
        factor = fct
        inp_size_single = (1, int(512*factor))
        model = dm.parallelTwoLayer(factor, repetable)
        opt_size = 512*fct
        
    elif model_name == "ParallelThreeLayer":
        factor = fct
        inp_size_single = (1, int(512*factor))
        model = dm.parallelThreeLayer(factor, repetable)
        opt_size = 512*fct
        
    elif model_name == "ShortLinear":
        factor = fct
        inp_size_single = (1, int(512*factor))
        model = dm.shortLinearModel(factor, repetable)
        opt_size = 512*fct
        
    elif model_name == "TallParallelModel":
        factor = fct
        inp_size_single = (1, int(512*factor))
        model = dm.tallParallelModel(factor, repetable)
        opt_size = 512*fct
        
    elif model_name == "LinearModel":
        factor = fct
        inp_size_single = (1, int(512*factor))
        model = dm.linearModel(factor, repetable)
        opt_size = 512*fct
        
    elif model_name == "OneLayer":
        factor = fct
        inp_size_single = (1, int(512*factor))
        model = dm.oneLayer(factor, repetable)
        opt_size = 512*fct

    # elif model_name == <model_name>:
    #   inp_size_single = <size of single input as tuple> # eg: for RGB image (3,300,300)
    #   model = import the model from models library
    #   opt_size = <length of output> # assumed to be 1D (also size of labels)
    #
    #   optionals:
    #   lr = 
    #   any other auxillary info -> model_info
    #   If specific learning rate or scaling for inpt is required -> lr, inpt_factor
    
    else:
        raise ValueError("model_name not valid!")

    return model, inp_size_single, opt_size, model_info, lr, inpt_factor