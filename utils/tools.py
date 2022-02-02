def _getactiv(activ):
    if activ == "relu":
        return nn.ReLU(inplace=True)
    elif activ == "gelu":
        return nn.GeLU(inplace=True)
    elif activ == "glu":
        return nn.GLU(inplace=True)
