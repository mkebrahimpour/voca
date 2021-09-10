'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.

You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.

Any use of the computer program without a valid license is prohibited and liable to prosecution.

Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about VOCA is available at http://voca.is.tue.mpg.de.
For comments or questions, please email us at voca@tue.mpg.de
'''


import os
import numpy as np
import tensorflow as tf

import torch
from utils.guo_model import VOCAModel
import scipy
from scipy.io import wavfile

from utils.audio_handler import  AudioHandler
from psbody.mesh import Mesh
from pytictoc import TicToc

def one_hot(x):
    x = np.expand_dims(x,-1)
    condition = torch.zeros(x.shape[0], 8)
    condotion = condition.scatter_(1, condition.type(torch.LongTensor), 1)
    return condition

def process_audio(ds_path, audio, sample_rate):
    config = {}
    config['deepspeech_graph_fname'] = ds_path
    config['audio_feature_type'] = 'deepspeech'
    config['num_audio_features'] = 29

    config['audio_window_size'] = 16
    config['audio_window_stride'] = 1

    tmp_audio = {'subj': {'seq': {'audio': audio, 'sample_rate': sample_rate}}}
    audio_handler = AudioHandler(config)
    return audio_handler.process(tmp_audio)['subj']['seq']['audio']


def output_sequence_meshes(sequence_vertices, template, out_path):
    out_path = os.path.join(out_path, 'meshes')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    num_frames = sequence_vertices.shape[0]
    for i_frame in range(num_frames):
        out_fname = os.path.join(out_path, '%05d.obj' % i_frame)
        Mesh(sequence_vertices[i_frame], template.f).write_obj(out_fname)


def inference(config, tf_model_fname, ds_fname, audio_fname, template_fname, condition_idx, out_path):
    template = Mesh(filename=template_fname)

    sample_rate, audio = wavfile.read(audio_fname)
    if audio.ndim != 1:
        print('Audio has multiple channels, only first channel is considered')
        audio = audio[:,0]

    model = VOCAModel(config)
    checkpoint = torch.load(tf_model_fname)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    processed_audio = process_audio(ds_fname, audio, sample_rate)

    # Load previously saved meta graph in the default graph


    num_frames = processed_audio.shape[0]
    world_size = 1
    rank = 0
    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))

#    model = VOCAModel(config)
#    checkpoint = torch.load(tf_model_fname)
#    model.load_state_dict(checkpoint['model_state_dict'])
    t = TicToc()
    t.tic()
    model = model.to(device_ids[0])
    
    print('model loaded')

    subject_idx = np.repeat(condition_idx-1, num_frames)
    condition = one_hot(subject_idx)
    condition = condition.to(device_ids[0])
    
    face_templates = np.repeat(template.v[np.newaxis, :, :], num_frames, axis=0)
#    processed_audio = np.expand_dims(np.stack(processed_audio), -1)
    processed_audio = torch.from_numpy(processed_audio).to(device_ids[0])
    processed_audio = processed_audio.type(torch.cuda.FloatTensor)
    feat = model.speech_encoder(processed_audio, condition)
    expression_offset = model.decoder(feat)
    expression_offset = expression_offset.float().cpu().detach().numpy()
    predicted = expression_offset + face_templates

    output_sequence_meshes(predicted, template, out_path)
    t.toc()

def inference_interpolate_styles(tf_model_fname, ds_fname, audio_fname, template_fname, condition_weights, out_path):
    template = Mesh(filename=template_fname)

    sample_rate, audio = wavfile.read(audio_fname)
    if audio.ndim != 1:
        print('Audio has multiple channels, only first channel is considered')
        audio = audio[:, 0]

    processed_audio = process_audio(ds_fname, audio, sample_rate)

    # Load previously saved meta graph in the default graph
    saver = tf.train.import_meta_graph(tf_model_fname + '.meta')
    graph = tf.get_default_graph()

    speech_features = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/speech_features:0')
    condition_subject_id = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/condition_subject_id:0')
    is_training = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/is_training:0')
    input_template = graph.get_tensor_by_name(u'VOCA/Inputs_decoder/template_placeholder:0')
    output_decoder = graph.get_tensor_by_name(u'VOCA/output_decoder:0')

    non_zeros = np.where(condition_weights > 0.0)[0]
    condition_weights[non_zeros] /= sum(condition_weights[non_zeros])

    num_frames = processed_audio.shape[0]
    output_vertices = np.zeros((num_frames, template.v.shape[0], template.v.shape[1]))

    with tf.Session() as session:
        # Restore trained model
        saver.restore(session, tf_model_fname)

        for condition_id in non_zeros:
            feed_dict = {speech_features: np.expand_dims(np.stack(processed_audio), -1),
                         condition_subject_id: np.repeat(condition_id, num_frames),
                         is_training: False,
                         input_template: np.repeat(template.v[np.newaxis, :, :, np.newaxis], num_frames, axis=0)}
            predicted_vertices = np.squeeze(session.run(output_decoder, feed_dict))
            output_vertices += condition_weights[condition_id] * predicted_vertices

        output_sequence_meshes(output_vertices, template, out_path)
