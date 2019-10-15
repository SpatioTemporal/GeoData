
# Read h5 file and try CCL.

# Clean up E1

import geodata as gd
import h5py as h5
from netCDF4 import Dataset
import numpy as np
import pystare as ps

import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs

import yaml

from stopwatch import sw_timer

import cv2
from ccl_marker_stack import ccl_marker_stack

###########################################################################
# Helper functions

def mnmx_masked(x,idx):
    return (np.min(x[idx]),np.max(x[idx]))

def scale_to_uint8(x,mnmx=None,offset=0):
    if mnmx is None:
        mn = np.nanmin(x)
        mx = np.nanmax(x)
        mnmx = (mn,mx)
    else:
        mn=mnmx[0]
        mx=mnmx[1]
    xs = offset+((255.0-offset)*(x-mn))
    if mx-mn != 0:
        xs=xs/(mx-mn)
    return np.uint8(xs),mnmx

###########################################################################
# https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python
# For fun viz.
def imshow_components(labels):
    # Map component labels to hue val
    # label_hue = np.uint8(179*labels/np.max(labels))
    label_hue = np.uint8(179*labels/64)
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    # cv2.imshow('labeled.png', labeled_img); cv2.waitKey()
    return labeled_img

def describe_var(msg,var):
    # print('var: ',type(var).__module__)
    if isinstance(var,(np.ndarray,np.generic)):
        print('%16s'%msg,' np.type,dtype,shape ',type(var),var.dtype,var.shape)
    else:
        if type(var) is list:
            print('%16s'%msg,' type,len            ',type(var),len(var))
        else:
            print('%16s'%msg,' type                ',type(var))
    return

###########################################################################
# Get the files ready

class ccl_tracker(object):
    def __init__(self,workFileNames,verbose=False):
        self.baseName='sketchE3'
        self.workFileNames=workFileNames
        self.current_file = 0
        self.verbose = verbose
        self.verbose_io = verbose
        self.bx_lo = 1000
        # self.b45_threshold = -1850 # for m2_threshold = ~20
        self.b45_tpw_scale  =    17.6
        self.b45_tpw_offset = -2196
        self.m2_threshold   =    50.0
        self.b45_threshold = self.b45_tpw_scale*self.m2_threshold + self.b45_tpw_offset
        self.substitute_m2_for_b45 = False # Test with m2 data instead of b45
        self.marker_stack = ccl_marker_stack(global_latlon_grid=False)
        self.viz = False
        self.viz0 = False
        self.save_slice = False
        return

    def init_threshold(self,m2_threshold):
        "Models b45 threshold from m2's value (kgom2)"
        self.m2_threshold  = m2_threshold
        self.b45_threshold = self.b45_tpw_scale*self.m2_threshold + self.b45_tpw_offset
        return

    def next(self):
        if self.current_file == len(self.workFileNames):
            return False

        workFileName = self.workFileNames[self.current_file]
        if self.verbose_io:
            print('loading ',workFileName)
        workFile = h5.File(workFileName,'r')

        tpw_scale  = workFile['/merra2_description']['tpw_scale']
        tpw_offset = workFile['/merra2_description']['tpw_offset']
        m2_img = tpw_offset + tpw_scale*workFile['/image']['merra2_tpw']

        # b3_img = workFile['/image']['goes_b3']
        b4_img = workFile['/image']['goes_b4']
        b5_img = workFile['/image']['goes_b5']

        nx = workFile['/image_description']['nx']
        ny = workFile['/image_description']['ny']

        # Mask to valid and shift data values to zero
        bx_lo = self.bx_lo
        idx_valid   = np.where((b5_img>bx_lo) & (b4_img>bx_lo))
        idx_invalid = np.where((b5_img<=bx_lo) & (b4_img<=bx_lo))
        b45_img = np.full(b5_img.shape,0,dtype=b5_img.dtype)
        # Shift to zero
        b45_img[idx_valid] = b4_img[idx_valid]-b5_img[idx_valid]-self.b45_tpw_offset # Shift to zero
        b45_threshold = self.b45_threshold - self.b45_tpw_offset # Shift to zero
        # Slam the negative
        b45_img[np.where(b45_img<0)] = 0 # Slam the negative. Sorry. Mea culpa.

        # Print summary to check
        if self.verbose:
            print('*** summary ***')
            print('m2_threshold = %f kgom2, b45_threshold = %f d-counts-zeroed, %f d-counts '%(self.m2_threshold,b45_threshold,b45_threshold+self.b45_tpw_offset))
            print('b45_img mnmx: ',np.amin(b45_img),np.amax(b45_img))
            print('***************')

        # Copied the following from ccl2d.

        if self.substitute_m2_for_b45:
            b45_img       = m2_img
            b45_threshold = 30.0

        if idx_valid is not None:
            data_scaled,b45_img_mnmx = scale_to_uint8(b45_img,mnmx_masked(b45_img,idx_valid)); 
        else:
            data_scaled,b45_img_mnmx = scale_to_uint8(b45_img)

        mnmx = b45_img_mnmx
        data = np.zeros(data_scaled.shape,dtype=np.uint8)
        data = data_scaled.reshape(ny,nx)
        d_trigger,_    = scale_to_uint8(b45_threshold,mnmx)
        d_out          = int(255)

        # Why does the limb show up in the thresh, but not the labels?

        # Eliminate the sky.
        # data[np.where(data < (255*3000/mx))] = 255

        if self.verbose:
            # print('d type: ',type(data))
            print('d trigger,out: ',d_trigger,d_out)
            print('d mnmx:  ',np.amin(data),np.amax(data))
            print('***************')

        if False:
            cv2.imshow('data',data); cv2.waitKey(0); cv2.destroyAllWindows()

        # This works
        if True:
            # Pass in data, ask for threshold
            # marker_stack = ccl_marker_stack(global_latlon_grid=False) 
            m0_new,m1_new,m0_eol,translation01\
                = self.marker_stack.make_slice_from(
                    data
                    ,(d_trigger,d_out)
                    ,graph=False
                    ,thresh_inverse=False
                    ,norm_data=False
                    ,perform_threshold=True
                    ,discard_below_pixel_area=10000
                )
            markers=m1_new
        if self.verbose:
            # print('markers type,len,shape,dtype ',type(markers),len(markers),markers.shape,markers.dtype)
            describe_var('m0_new',m0_new)
            describe_var('m1_new',m1_new)
            describe_var('m0_eol',m0_eol)
            describe_var('translation01',translation01)
            describe_var('markers',markers)
            # print('markers ',markers)

        thresh = None

        # The following two also work
        if False:
            # Pass in external threshold
            # ret,thresh  = cv2.threshold(data,d_trigger,d_out,cv2.THRESH_BINARY_INV) # less than, for b5
            ret,thresh  = cv2.threshold(data,d_trigger,d_out,cv2.THRESH_BINARY) # less than, for b5
            if verbose:
                print('thresh ret:  ',type(ret),ret)
                print('thresh type: ',type(thresh),thresh.shape,np.amin(thresh),np.amax(thresh))
            # marker_stack = ccl_marker_stack(global_latlon_grid=False) 
            m0_new,m1_new,m0_eol,translation01\
                = self.marker_stack.make_slice_from(
                    thresh
                    ,(d_trigger,d_out)
                    ,graph=False
                    ,thresh_inverse=False
                    ,norm_data=False
                ,perform_threshold=False)
            markers=m1_new
            if verbose:
                print('markers type,len ',type(markers),len(markers))
                # print('markers ',markers)

        # Save a CCL2D result to an HDF file.
        if self.save_slice:
            outFileName = self.baseName+"."+".".join(workFileName.split('.')[:-1])+'-markers-1.h5'
            if self.verbose_io:
                print('writing ',outFileName)
            outFile = h5.File(outFileName,'w')
            
            translation01_yaml = yaml.dump(translation01)
            t_yaml_len = len(translation01_yaml)

            markers_dtype = np.dtype([
                ('markers',markers.dtype)
            ])
            markers_ds = outFile.create_dataset('markers',[markers.size],dtype=markers_dtype)

            markers_description_dtype = np.dtype([
                ('nx',np.int)
                ,('ny',np.int)
            ])
            markers_description_ds = outFile.create_dataset('markers_description',[],dtype=markers_description_dtype)

            translation_dtype = np.dtype([
                ('translation01_yaml','S%i'%t_yaml_len)
            ])
            translation_yaml_ds = outFile.create_dataset('translation_yaml',[],dtype=translation_dtype)

            outFile['/markers']['markers']                     = markers[:,:].flatten()
            outFile['/markers_description']['nx']              = markers.shape[1]
            outFile['/markers_description']['ny']              = markers.shape[0]
            outFile['/translation_yaml']['translation01_yaml'] = translation01_yaml
            outFile.close()

            if self.viz:
                print('reading ',outFileName)
                inFile = h5.File(outFileName,'r')
                tmp_Marker = inFile['/markers']['markers']
                tmp_nx     = inFile['/markers_description']['nx']
                tmp_ny     = inFile['/markers_description']['ny']
                inFile.close()
                nrows = 2
                ncols = 2
                fig,axs = plt.subplots(nrows=nrows,ncols=ncols)
                for irow in range(nrows):
                    for icol in range(ncols):
                        axs[irow,icol].get_xaxis().set_visible(False)
                        axs[irow,icol].get_yaxis().set_visible(False)
                irow,icol = (0,0)
                axs[irow,icol].set_title('m2_img')
                axs[irow,icol].imshow(m2_img.reshape(ny,nx))    
                irow,icol = (0,1)
                axs[irow,icol].set_title('b45_img')
                axs[irow,icol].imshow(b45_img.reshape(ny,nx))    
                irow,icol = (1,0)
                axs[irow,icol].set_title('inFile markers')
                axs[irow,icol].imshow(imshow_components(tmp_Marker.reshape(tmp_ny,tmp_nx)))
                irow,icol = (1,1)
                axs[irow,icol].set_title('markers')
                axs[irow,icol].imshow(imshow_components(markers.reshape(tmp_ny,tmp_nx)))
                plt.show()

        if False:
            cv2.imshow('thresh',thresh); cv2.waitKey(0); cv2.destroyAllWindows()

        if False:
            ret,markers = cv2.connectedComponents(thresh)

        markers_mx = np.amax(markers)
        if self.verbose:
            print('markers_mx: ',markers_mx)

        if markers_mx == 0:
            markers_mx = 1

        data1=markers.astype(np.float)/markers_mx
        if self.verbose:
            print('markers',type(markers),type(data1))

        if False:
            cv2.imshow('markers',data1); cv2.waitKey(0); cv2.destroyAllWindows()

        if self.viz0:
            print('unique markers: ',np.unique(markers.flatten()).size)

            nrows = 3
            ncols = 2
            fig,axs = plt.subplots(nrows=nrows,ncols=ncols)

            for irow in range(nrows):
                for icol in range(ncols):
                    axs[irow,icol].get_xaxis().set_visible(False)
                    axs[irow,icol].get_yaxis().set_visible(False)

            irow,icol = (0,0)
            axs[irow,icol].imshow(b45_img.reshape(ny,nx))
            irow,icol = (1,0)
            axs[irow,icol].imshow(data)
            if thresh is not None:
                irow,icol = (2,0)
                axs[irow,icol].imshow(thresh)
            irow,icol = (0,1)
            axs[irow,icol].imshow(m2_img.reshape(ny,nx))
            irow,icol = (1,1)
            axs[irow,icol].imshow(markers)
            irow,icol = (2,1)
            axs[irow,icol].imshow(imshow_components(markers))
            plt.show()

        if self.viz:
            nrows = 2
            ncols = 2
            fig,axs = plt.subplots(nrows=nrows,ncols=ncols)
            for irow in range(nrows):
                for icol in range(ncols):
                    axs[irow,icol].get_xaxis().set_visible(False)
                    axs[irow,icol].get_yaxis().set_visible(False)
            irow,icol = (0,0)
            axs[irow,icol].imshow(m2_img.reshape(ny,nx))
            irow,icol = (1,0)
            axs[irow,icol].imshow(imshow_components(markers))
            irow,icol = (1,1)
            axs[irow,icol].imshow(b45_img.reshape(ny,nx))
            plt.show()

        self.current_file = self.current_file + 1
        return True

    def resolve(self):
        return self.marker_stack.resolve_labels_across_stack()

###########################################################################
# TODO Save list of source files to the output file.
    def save_resolution(self):
        print('res len: ',len(self.marker_stack.m_results_translated))
        sw_timer.stamp('save-resolution-start')
        k=0
        for workFileName in self.workFileNames:
            outFileName = self.baseName+"."+".".join(workFileName.split('.')[:-1])+'-final-markers-1.h5'
            if self.verbose_io:
                print('writing ',outFileName)
            outFile = h5.File(outFileName,'w')
            
            # Presume we've already called the resolution routine.
            markers = self.marker_stack.m_results_translated[k]
            
            # read marker from file, or save from the resolution?

            markers_dtype = np.dtype([
                ('markers',markers.dtype)
            ])
            markers_ds = outFile.create_dataset('markers',[markers.size],dtype=markers_dtype)

            markers_description_dtype = np.dtype([
                ('nx',np.int)
                ,('ny',np.int)
            ])
            markers_description_ds = outFile.create_dataset('markers_description',[],dtype=markers_description_dtype)

            outFile['/markers']['markers']                     = markers[:,:].flatten()
            outFile['/markers_description']['nx']              = markers.shape[1]
            outFile['/markers_description']['ny']              = markers.shape[0]
            outFile.close()
            k=k+1
        sw_timer.stamp('save-resolution-end')

if __name__ == '__main__':

    sw_timer.stamp('main start')

    workFileNames = [
        'sketchF1.0x007d5c684080008a.h5'
        ,'sketchF1.0x007d5c685e80008a.h5'
        ,'sketchF1.0x007d5c688080008a.h5'
        ,'sketchF1.0x007d5c689e80008a.h5' ]
    
    tracker = ccl_tracker(workFileNames)
    tracker.save_slice = True
    tracker.verbose    = False
    tracker.verbose_io = True
    tracker.viz0       = False
    while tracker.next():
        print('iter: ',tracker.current_file,' time (s): ',sw_timer.delta_since_start())
    tracker.resolve()
    tracker.save_resolution()
    sw_timer.stamp('main end')

    print(sw_timer.report_all())
