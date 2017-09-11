# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 11:28:53 2016

@author: bschloss
"""
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
from nipype.interfaces.utility import IdentityInterface as ID
from nipype.pipeline.engine import Workflow, Node, MapNode
import os

orig_dir = '/gpfs/group/pul8/default/read/'
par_dir = orig_dir + '002/'
dti_preproc_dir = par_dir + 'DTI_Preprocessing/'

fsamples = dti_preproc_dir + 'BEDPOSTX_Merged_Fsamples/'
fsamples += os.listdir(fsamples)[0] + '/'
fsamples = [fsamples + f for f in os.listdir(fsamples)]

mask = dti_preproc_dir + 'Nodif_Brain_Mask/'
mask += os.listdir(mask)[0] + '/'
mask = [mask + f for f in os.listdir(mask)]

phsamples = dti_preproc_dir + 'BEDPOSTX_Merged_PHsamples/'
phsamples += os.listdir(phsamples)[0] + '/'
phsamples = [phsamples + f for f in os.listdir(phsamples)]

seedr = orig_dir + 'DTI_Masks/Uncinate_Masks/subject_template_mask_post_R.nii.gz'
seedl = orig_dir +  'DTI_Masks/Uncinate_Masks/subject_template_mask_post_L.nii.gz'

thsamples = dti_preproc_dir + 'BEDPOSTX_Merged_THsamples/'
thsamples += os.listdir(thsamples)[0] + '/'
thsamples = [thsamples + f for f in os.listdir(thsamples)]

waypointr = orig_dir + 'Uncinate_Masks/subject_template_mask_ant_R.nii.gz'
waypointl = orig_dir + 'Uncinate_Masks/subject_template_mask_ant_L.nii.gz'

terminationr = orig_dir + 'Uncinate_Masks/R_Inf_Front_Mask.nii.gz'
terminationl = orig_dir + 'Uncinate_Masks/L_Inf_Front_Mask.nii.gz'

xfm = dti_preproc_dir + 'MNI2D_Warp_Field_File/'
xfm += os.listdir(xfm)[0] + '/'
xfm = [xfm + f for f in os.listdir(xfm)]

inv_xfm = dti_preproc_dir + 'D2MNI_Warp_Field_File/'
inv_xfm += os.listdir(inv_xfm)[0] + '/'
inv_xfm = [inv_xfm + f for f in os.listdir(inv_xfm)]

#Define Workflow
preproc = Workflow(name = "Preprocessing")#,base_dir='/media/bschloss/Extra Drive 1/Preprocessing/')

#Use DataGrabber function to get all of the data
data = Node(ID(fields=['fsamples','mask','phsamples','seedr','seedl','thsamples','waypointr',
                       'waypointl','xfm','inv_xfm','terminationl','terminationr'],mandatory_inputs=False),name='Data')
data.inputs.fsamples = [fsamples]
data.inputs.mask = mask
data.inputs.phsamples = [phsamples]
data.inputs.seedr = seedr
data.inputs.seedl = seedl
data.inputs.thsamples = [thsamples]
data.inputs.waypointr = waypointr
data.inputs.waypointl = waypointl
data.inputs.xfm = xfm
data.inputs.inv_xfm = inv_xfm
data.inputs.terminationr = terminationr
data.inputs.terminationl = terminationl

#Use the DataSink function to store all outputs
datasink = Node(nio.DataSink(), name= 'Ouput')
datasink.inputs.base_directory = par_dir + 'DTI_Preprocessing/'

probtrackx2_uf_s2t_r = MapNode(fsl.ProbTrackX2(onewaycondition=True,
                                                     loop_check=True,
                                                     sample_random_points=False),
                     name = "ProbTrackX2_S2TR",
                     iterfield = ['fsamples',
                                  'mask',
                                  'phsamples',
                                  'thsamples',
                                  'xfm',
                                  'inv_xfm'])
preproc.connect(data,'fsamples',probtrackx2_uf_s2t_r,'fsamples')
preproc.connect(data,'mask',probtrackx2_uf_s2t_r,'mask')
preproc.connect(data,'phsamples',probtrackx2_uf_s2t_r,'phsamples')
preproc.connect(data,'seedr',probtrackx2_uf_s2t_r,'seed')
preproc.connect(data,'thsamples',probtrackx2_uf_s2t_r,'thsamples')
preproc.connect(data,'waypointr',probtrackx2_uf_s2t_r,'waypoints')
preproc.connect(data,'xfm',probtrackx2_uf_s2t_r,'xfm')
preproc.connect(data,'inv_xfm',probtrackx2_uf_s2t_r,'inv_xfm')
#preproc.connect(data,'terminationr',probtrackx2_uf_s2t_r,'stop_mask')
preproc.connect(probtrackx2_uf_s2t_r,'fdt_paths',datasink,'UF_FP2TP_R')

probtrackx2_uf_t2s_r = MapNode(fsl.ProbTrackX2(onewaycondition=True,
                                                     loop_check=True,
                                                     sample_random_points=False),
                     name = "ProbTrackX2_T2SR",
                     iterfield = ['fsamples',
                                  'mask',
                                  'phsamples',
                                  'thsamples',
                                  'xfm',
                                  'inv_xfm'])
preproc.connect(data,'fsamples',probtrackx2_uf_t2s_r,'fsamples')
preproc.connect(data,'mask',probtrackx2_uf_t2s_r,'mask')
preproc.connect(data,'phsamples',probtrackx2_uf_t2s_r,'phsamples')
#preproc.connect(data,'terminationr',probtrackx2_uf_t2s_r,'seed')
preproc.connect(data,'thsamples',probtrackx2_uf_t2s_r,'thsamples')
preproc.connect(data,'waypointr',probtrackx2_uf_t2s_r,'waypoints')
preproc.connect(data,'xfm',probtrackx2_uf_t2s_r,'xfm')
preproc.connect(data,'inv_xfm',probtrackx2_uf_t2s_r,'inv_xfm')
preproc.connect(data,'seedr',probtrackx2_uf_t2s_r,'stop_mask')
preproc.connect(probtrackx2_uf_t2s_r,'fdt_paths',datasink,'UF_TP2FP_R')

probtrackx2_uf_s2t_l = MapNode(fsl.ProbTrackX2(onewaycondition=True,
                                                     loop_check=True,
                                                     sample_random_points=False),
                     name = "ProbTrackX2_S2TL",
                     iterfield = ['fsamples',
                                  'mask',
                                  'phsamples',
                                  'thsamples',
                                  'xfm',
                                  'inv_xfm'])
preproc.connect(data,'fsamples',probtrackx2_uf_s2t_l,'fsamples')
preproc.connect(data,'mask',probtrackx2_uf_s2t_l,'mask')
preproc.connect(data,'phsamples',probtrackx2_uf_s2t_l,'phsamples')
preproc.connect(data,'seedl',probtrackx2_uf_s2t_l,'seed')
preproc.connect(data,'thsamples',probtrackx2_uf_s2t_l,'thsamples')
preproc.connect(data,'waypointl',probtrackx2_uf_s2t_l,'waypoints')
preproc.connect(data,'xfm',probtrackx2_uf_s2t_l,'xfm')
preproc.connect(data,'inv_xfm',probtrackx2_uf_s2t_l,'inv_xfm')
#preproc.connect(data,'terminationl',probtrackx2_uf_s2t_l,'stop_mask')
preproc.connect(probtrackx2_uf_s2t_l,'fdt_paths',datasink,'UF_FP2TP_L')

probtrackx2_uf_t2s_l = MapNode(fsl.ProbTrackX2(onewaycondition=True,
                                                     loop_check=True,
                                                     sample_random_points=False),
                     name = "ProbTrackX2_T2SL",
                     iterfield = ['fsamples',
                                  'mask',
                                  'phsamples',
                                  'thsamples',
                                  'xfm',
                                  'inv_xfm'])
preproc.connect(data,'fsamples',probtrackx2_uf_t2s_l,'fsamples')
preproc.connect(data,'mask',probtrackx2_uf_t2s_l,'mask')
preproc.connect(data,'phsamples',probtrackx2_uf_t2s_l,'phsamples')
#preproc.connect(data,'terminationl',probtrackx2_uf_t2s_l,'seed')
preproc.connect(data,'thsamples',probtrackx2_uf_t2s_l,'thsamples')
preproc.connect(data,'waypointl',probtrackx2_uf_t2s_l,'waypoints')
preproc.connect(data,'xfm',probtrackx2_uf_t2s_l,'xfm')
preproc.connect(data,'inv_xfm',probtrackx2_uf_t2s_l,'inv_xfm')
preproc.connect(data,'seedl',probtrackx2_uf_t2s_l,'stop_mask')
preproc.connect(probtrackx2_uf_t2s_l,'fdt_paths',datasink,'UF_TP2FP_L')

#preproc.write_graph(dotfilename='DTI_Preprocessing_Graph',format='svg')
#preproc.write_graph(dotfilename='DTI_Preprocessing_Graph',format='svg',graph2use='exec')
preproc.run(plugin='MultiProc', plugin_args={'n_procs' : 4})
