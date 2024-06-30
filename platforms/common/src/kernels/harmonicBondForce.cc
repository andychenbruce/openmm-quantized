float2 bondParams = PARAMS[index];
real deltaIdeal = r-(real)bondParams.x;
energy += (real)0.5 * (real)bondParams.y*deltaIdeal*deltaIdeal;
real dEdR = (real)bondParams.y * deltaIdeal;
