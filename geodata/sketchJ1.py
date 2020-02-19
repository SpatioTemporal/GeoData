
import numpy as np
from geodata import id_from_ij, ij_from_id, make_id_idx, id_fixedwidth_from_ij, ij_from_id_fixedwidth, make_id_fixedwidth_idx, id_from_id_fixedwidth, id_fixedwidth_from_id

## def id_from(i,j,shape):
##     "Return usual linear index into 2D array, where shape = [Nj,Ni] and array is indexed as a[j,i]"
##     return i+j*shape[1]
## 
## def from_id(id,shape):
##     return id % shape[1],int(id/shape[1])
## 
## def make_id_idx(shape):
##     idx = np.zeros(shape,dtype=np.int64)
##     for j in np.arange(shape[0]):
##         for i in np.arange(shape[1]):
##             idx[j,i] = id_from(i,j,shape)
##     return idx
## 
## def id1_from(ix,jy,shape):
##     "Packs the two array index values into a single 64-bit integer. Note: a little sloppy in type decl."
##     return (jy << 32)+ix;
## 
## def from_id1(id1):
##     "Unpack two array index values from a single 64-bit integer. Note we're making assumptions about types."
##     return (id1 & ((1 << 32)-1)),(id1 >> 32)
## 
## def make_id1_idx(shape):
##     idx = np.zeros(shape,dtype=np.int64)
##     for j in np.arange(shape[0]):
##         for i in np.arange(shape[1]):
##             idx[j,i] = id1_from(i,j,shape)
##     return idx
            


def main():
    print('hello')

    nx = 5
    ny = 3
    sh = [ny,nx]

    x = np.linspace(-180,180,nx)
    y = np.linspace(-90,90,ny)
    xg,yg = np.meshgrid(x,y)
    
    xg_flat = xg.flatten()
    yg_flat = yg.flatten()

    i = np.arange(nx)
    j = np.arange(ny)
    ig,jg = np.meshgrid(i,j)
    ig_flat = ig.flatten()
    jg_flat = jg.flatten()
    idx_ijg = np.zeros(sh,dtype=np.int)
    for i_ in i:
        for j_ in j:
            # print('ij: ',i_,j_)
            idx_ijg[j_,i_] = id_from_ij(i_,j_,sh)

    idx_ijg_flat = idx_ijg.flatten()
    idx_ig = np.zeros([idx_ijg.size],dtype=np.int64)
    idx_jg = np.zeros([idx_ijg.size],dtype=np.int64)
    for k in np.arange(idx_ijg.size):
        i_,j_ = ij_from_id(idx_ijg_flat[k],sh)
        idx_ig[k] = i_
        idx_jg[k] = j_

    idx1_ijg = np.zeros(sh,dtype=np.int64)
    for i_ in i:
        for j_ in j:
            # print('ij: ',i_,j_)
            idx1_ijg[j_,i_] = id_fixedwidth_from_ij(i_,j_,sh)
            print('ij,idx: ',i_,j_,idx1_ijg[j_,i_])

    idx1_ijg_flat = idx1_ijg.flatten()
    print('idx1_ijg_flat:       ',idx1_ijg_flat)
    # print('idx1_ijg_flat: size: ',idx1_ijg_flat.size)
    idx1_ig_inv  = np.zeros([idx1_ijg_flat.size],dtype=np.int64)
    idx1_jg_inv  = np.zeros([idx1_ijg_flat.size],dtype=np.int64)
    for k in range(idx1_ijg_flat.size):
        print(k,' idx ',idx1_ijg_flat[k])
        i_,j_ = ij_from_id_fixedwidth(idx1_ijg_flat[k],sh)
        print(k,'i_,j_',i_,j_)
        idx1_ig_inv[k] = i_
        idx1_jg_inv[k] = j_

    a=np.zeros(sh,dtype=np.int)
    src_coord = np.arange(a.size)
    coord_g = src_coord.reshape(sh)

    src_coord_fixed     = np.array([id_fixedwidth_from_id(id,sh) for id  in src_coord])
    src_coord_fixed_inv = np.array([id_from_id_fixedwidth   (id1,sh)   for id1 in src_coord_fixed])

    print('x\n',x)
    print('y\n',y)

    print('xg\n',xg)
    print('yg\n',yg)

    print('xg_flat\n',xg_flat)
    print('yg_flat\n',yg_flat)
    print('src_coord\n',src_coord)

    print('coord_g\n',coord_g)

    print('i\n',i)
    print('j\n',j)

    print('ig sh=%s\n'%str(ig.shape),ig)
    print('jg sh=%s\n'%str(jg.shape),jg)
    print('ig_flat\n',ig_flat)
    print('jg_flat\n',jg_flat)

    print('idx_ijg\n',idx_ijg)
    print('idx_ijg.flatten()\n',idx_ijg.flatten())
    print('idx_ijg_flat\n',idx_ijg_flat)
    print('make_id_idx\n',make_id_idx(sh))

    print('idx1_ijg.flatten()\n',idx1_ijg.flatten())

    print('idx1_ig_inv\n',idx1_ig_inv)
    print('idx1_jg_inv\n',idx1_jg_inv)
    print('make_id1_idx\n',make_id_fixedwidth_idx(sh))

    print('\nsrc_coord\n',src_coord)
    print('src_coord_fixed\n',src_coord_fixed)
    print('src_coord_fixed array\n',id_fixedwidth_from_id(src_coord,sh))
    print('src_coord_fixed_inv\n',src_coord_fixed_inv)
    print('src_coord_fixed_inv array\n',id_from_id_fixedwidth(src_coord_fixed,sh))

    return


if __name__ == '__main__':
    main()
