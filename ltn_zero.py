import click as ck
import csv

import torch
from tqdm import tqdm
import gc

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import math

import torch as th
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR

import ltn

from torch_utils import FastTensorDataLoader
from utils import Ontology

from ltn_zero_utils import Forall, Exists, And, Implies, sparse_dense_mul, sparse_repeat


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
@ck.option(
    '--batch-size', '-bs', default=37,
    help='Batch size for training')
@ck.option(
    '--axiom-batch-size', '-abs', default=1000,
    help='Axiom batch size for training')
@ck.option(
    '--epochs', '-ep', default=256,
    help='Training epochs')
@ck.option(
    '--latent-dim', '-ld', default=128,
    help='Latent dimension')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--device', '-d', default='cuda',
    help='Device')
def main(data_root, ont, batch_size, axiom_batch_size, epochs, latent_dim, load, device):
    go_file = f'{data_root}/go.norm'
    model_file = f'{data_root}/{ont}/deepgoltn_zero_10.th'
    # terms_file = f'{data_root}/{ont}/terms.pkl'
    terms_file = f'{data_root}/{ont}/terms_zero_10.pkl'
    out_file = f'{data_root}/{ont}/predictions_deepgoltn_zero_10.pkl'

    go = Ontology(f'{data_root}/go.obo', with_rels=True)
    loss_func = nn.BCELoss()
    iprs_dict, terms_dict, train_data, valid_data, test_data, test_df = load_data(data_root, ont, terms_file)
    n_terms = len(terms_dict)
    n_iprs = len(iprs_dict)

    # load normal forms
    nf1, nf2, nf3, nf4, relations, zero_classes = load_normal_forms(
        go_file, terms_dict)

    # add hasFunction relation to relations dict
    relations['hasFunction'] = len(relations)

    n_rels = len(relations)
    n_zeros = len(zero_classes)
    print('n_zeros', n_zeros)

    total_GO_class_ids = list(terms_dict.values()) + list(zero_classes.values())
    n_GO_classes = len(total_GO_class_ids)

    train_features, train_labels = train_data
    valid_features, valid_labels = valid_data
    test_features, test_labels = test_data

    # Train autoencoder for dimensionality reduction
    autoencoder = ConceptAutoencoder(n_iprs, latent_dim=latent_dim).to(device)
    loss_func = th.nn.MSELoss()
    autoencoder_load = True
    autoencoder_model_file = f'{data_root}/{ont}/autoenc_deepgoltn_zero_10.th'
    if not autoencoder_load:
        ae_batch_size = 10000
        train_loader = FastTensorDataLoader(
            *train_data, batch_size=ae_batch_size, shuffle=True)

        autoencoder_optimizer = th.optim.Adam(autoencoder.parameters(), lr=1e-3)

        print('Training autoencoder...')
        for epoch in range(10):
            autoencoder.train()
            train_loss = 0
            train_steps = int(math.ceil(len(train_labels) / ae_batch_size))
            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for batch_features, _ in train_loader:
                    bar.update(1)

                    batch_features = batch_features.to(device)
                    preds = autoencoder(batch_features)
                    loss = loss_func(preds, batch_features)

                    autoencoder_optimizer.zero_grad()

                    loss.backward()
                    autoencoder_optimizer.step()
                    train_loss += loss.detach().item()

            train_loss /= len(train_labels)

            print(f'Epoch {epoch}: Loss - {train_loss}')
        th.save(autoencoder.encoder.state_dict(), autoencoder_model_file)
    else:
        autoencoder.encoder.load_state_dict(th.load(autoencoder_model_file))
        print('BUmm')

    encoder = autoencoder.encoder

    # free memory after training autoencoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    gc.collect()
    th.cuda.empty_cache()

    nf1 = th.LongTensor(nf1).to(device)
    nf2 = th.LongTensor(nf2).to(device)
    nf3 = th.LongTensor(nf3).to(device)
    nf4 = th.LongTensor(nf4).to(device)
    normal_forms = nf1, nf2, nf3, nf4

    for i, nf in enumerate([nf1, nf2, nf3, nf4]):
        print(f'Num constraints nf{i+1}: {len(nf)=}')

    # initialize LTN constants for classes and relations
    concept_constants = F.one_hot(th.arange(0, n_GO_classes))
    # concept_constants = [ltn.Constant(one_hot_matrix[i, :]) for i in range(n_GO_classes)]
    rel_constants = F.one_hot(th.arange(0, n_rels))
    # rel_constants = [ltn.Constant(one_hot_matrix[i, :]) for i in range(n_rels)]

    # initialize models and respective predicates
    concept_net = DGLTNConceptModel(latent_dim, n_GO_classes, device).to(device)
    # C_go = ConceptLogitsToPredicate(concept_net)
    relation_net = DGLTNRelationModel(latent_dim, n_rels, device).to(device)
    # R_go = ConceptLogitsToPredicate(relation_net)
    skolemizer = Skolemizer(input_dim=latent_dim, nb_individuals=n_GO_classes, nb_rels=n_rels)
    embedding_model = IndividualEmbedder(n_GO_classes=n_GO_classes,
                                         latent_dim=latent_dim)

    concept_optimizer = th.optim.Adam(concept_net.parameters(), lr=1e-4)
    relation_optimizer = th.optim.Adam(relation_net.parameters(), lr=1e-4)
    skolemizer_optimizer = th.optim.Adam(skolemizer.parameters(), lr=1e-4)
    embedding_model_optimizer = th.optim.Adam(embedding_model.parameters(), lr=1e-4)

    # concept_scheduler = MultiStepLR(concept_optimizer, milestones=[1, 3,], gamma=0.1)
    # relation_scheduler = MultiStepLR(relation_optimizer, milestones=[1, 3,], gamma=0.1)

    train_loader = FastTensorDataLoader(
        *train_data, batch_size=batch_size, shuffle=True)
    valid_loader = FastTensorDataLoader(
        *valid_data, batch_size=batch_size, shuffle=False)
    test_loader = FastTensorDataLoader(
        *test_data, batch_size=batch_size, shuffle=False)

    valid_labels = valid_labels.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()

    normal_form_matrix_dict = build_normal_form_matrices(nb_GO_classes=n_GO_classes, normal_forms=normal_forms)

    index_tensor = th.arange(n_GO_classes).view(-1, 1)
    index_data_loader = FastTensorDataLoader(index_tensor, batch_size=batch_size, shuffle=True)

    p = 2
    best_loss = math.inf
    if not load:
        print('Training the model')
        log_file = open(f'{data_root}/train_logs.tsv', 'w')
        logger = csv.writer(log_file, delimiter='\t')
        for epoch in range(epochs):
            concept_net.train()
            relation_net.train()
            skolemizer.train()

            train_steps = int(math.ceil(len(train_labels) / batch_size))

            # @TODO calculate the following things
            # 0. calculate satisfaction of GO classes for first two normal forms
            # 1. calculate satisfaction of GO classes for second two normal forms
            # 2. include membership of proteins to respective GO classes with p \sqsubseteq \Exists R.C

            train_loss = 0

            # For implementation of more Fuzzy operators refer to https://github.com/bmxitalia/LTNtorch
            # Satisfaction of nf1 axioms:
            # \Forall x: C(x) \implies D(x)
            # nf1_matrix_c.size() = nf1_matric_d.size() = n_GO_classes x n_constraints
            nf1_matrix_c, nf1_matrix_d = normal_form_matrix_dict['nf1']
            num_constraints = nf1_matrix_c.size(1)
            index_vec = th.arange(num_constraints)

            for batch_indices in index_data_loader:
                # batch_features = batch_features[0]
                batch_indices = batch_indices.to(device)
                batch_features = embedding_model(batch_indices)

                X_GO = concept_net(batch_features)

                for i in range(0, num_constraints, axiom_batch_size):
                    batch_nf1_matrix_c = nf1_matrix_c.index_select(dim=1, index=index_vec[i:i+axiom_batch_size])
                    batch_nf1_matrix_d = nf1_matrix_d.index_select(dim=1, index=index_vec[i:i+axiom_batch_size])

                    X_C = th.sparse.mm(X_GO, batch_nf1_matrix_c)
                    X_D = th.sparse.mm(X_GO, batch_nf1_matrix_d)

                    out = Implies(X_C, X_D)  # Reichenbach Fuzzy Implication
                    # out = th.where(th.le(X_C, X_D), th.ones_like(X_C), X_D)  # Gödel Implication
                    out = Forall(out, dim=0, p=p) / batch_size  # \Forall x
                    out = Forall(out, dim=0, p=p) / num_constraints  # \Forall constraints
                    loss = 1 - out
                    concept_optimizer.zero_grad()
                    loss.backward()
                    concept_optimizer.step()

                    train_loss += loss.detach().item()

            # Satisfaction of nf2 axioms:
            # \Forall x: C(x) \sqcap D(x) \implies E(x)
            # nf2_matrix_c/d/e.size() = n_GO_classes x n_constraints
            nf2_matrix_c, nf2_matrix_d,  nf2_matrix_e = normal_form_matrix_dict['nf2']
            num_constraints = nf2_matrix_c.size(1)
            index_vec = th.arange(num_constraints)

            for batch_indices in index_data_loader:
                # batch_features = batch_features[0]
                batch_indices = batch_indices.to(device)
                batch_features = embedding_model(batch_indices)

                X_GO = concept_net(batch_features)

                for i in range(0, num_constraints, axiom_batch_size):
                    batch_nf2_matrix_c = nf2_matrix_c.index_select(dim=1, index=index_vec[i:i+axiom_batch_size])
                    batch_nf2_matrix_d = nf2_matrix_d.index_select(dim=1, index=index_vec[i:i+axiom_batch_size])
                    batch_nf2_matrix_e = nf2_matrix_e.index_select(dim=1, index=index_vec[i:i+axiom_batch_size])

                    X_C = th.sparse.mm(X_GO, batch_nf2_matrix_c)
                    X_D = th.sparse.mm(X_GO, batch_nf2_matrix_d)
                    X_E = th.sparse.mm(X_GO, batch_nf2_matrix_e)

                    X_C_AND_D = And(X_C, X_D)  # \sqcap^\mathcal{I} = *
                    out = Implies(X_C_AND_D, X_E)  # Reichenbach Fuzzy Implication
                    # out = th.where(th.le(X_C_AND_D, X_E), th.ones_like(X_C_AND_D), X_E)  # Gödel Implication
                    out = Forall(out, dim=0, p=p) / batch_size  # \Forall x
                    out = Forall(out, dim=0, p=p) / num_constraints  # \Forall constraints
                    loss = 1 - out
                    concept_optimizer.zero_grad()
                    loss.backward()
                    concept_optimizer.step()

                    train_loss += loss.detach().item()

            # Satisfaction of nf3 axioms:
            # R some C subClassOf D = \Forall x: R(x,f_R(x)) \sqcup C(f_R(x)) \Implies D(x)
            # nf3_matrix_c/d.size() = n_concepts x n_constraints, Sparse matrix
            # nf3_matrix_R.size() = n_rels x n_constraints, Sparse matrix
            nf3_matrix_R, nf3_matrix_c, nf3_matrix_d = normal_form_matrix_dict['nf3']
            num_constraints = nf3_matrix_c.size(1)
            index_vec = th.arange(num_constraints)

            for batch_indices in index_data_loader:
                # batch_features = batch_features[0]
                batch_indices = batch_indices.to(device)
                batch_features = embedding_model(batch_indices)

                X_GO = concept_net(batch_features)

                f_R_mat = skolemizer(batch_features)  # |B| x n_rels x n_individuals
                i_max = f_R_mat.argmax(dim=2)  # get individuals with highest values i.e. argmax, |B| x n_rels
                sat_max = Exists(f_R_mat, dim=2,
                                 p=3)  # calculate Exists along individuals, see utils for more info, |B| x n_rels

                i_max_embedded = embedding_model(i_max)  # |B| x n_rels x latent_dim
                X_f_R = concept_net(i_max_embedded)  # |B| x n_rel x n_GO_classes

                X_R_x_f_R = relation_net(batch_features.unsqueeze(1).repeat(1, n_rels, 1),
                                         i_max_embedded).view(batch_size, -1)  # |B| x n_rels

                for i in range(0, num_constraints, axiom_batch_size):
                    batch_nf3_matrix_R = nf3_matrix_R.index_select(dim=1, index=index_vec[i:i + axiom_batch_size])
                    batch_nf3_matrix_c = nf3_matrix_c.index_select(dim=1, index=index_vec[i:i + axiom_batch_size])
                    batch_nf3_matrix_d = nf3_matrix_d.index_select(dim=1, index=index_vec[i:i + axiom_batch_size])

                    X_D = th.sparse.mm(X_GO, batch_nf3_matrix_d)  # |B| x axiom_batch_size

                    X_R = th.sparse.mm(X_R_x_f_R, batch_nf3_matrix_R)  # |B| x axiom_batch_size
                    sat_max = th.sparse.mm(sat_max, batch_nf3_matrix_R)  # |B| x axiom_batch_size
                    X_R = X_R * sat_max  # include satisfaction from skolemizer

                    X_C = th.sparse.mm(X_f_R, batch_nf3_matrix_c)  # |B| x n_rel x axiom_batch_size
                    X_C = sparse_dense_mul(sparse_repeat(batch_nf3_matrix_R.unsqueeze(0), reps=batch_size, dim=0), X_C)
                    X_C = X_C.sum(dim=1).to_dense()  # |B| x axiom_batch_size

                    X_R_some_C = And(X_R, X_C)  # product t-norm through pointwise multiplication
                    out = Implies(X_R_some_C, X_D)  # Reichenbach Fuzzy Implication

                    out = ((out ** p).sum(dim=0) ** (1 / p)) / batch_size  # \Forall x
                    out = ((out ** p).sum() ** (1 / p)) / num_constraints  # \Forall constraints
                    loss = 1 - out
                    concept_optimizer.zero_grad()
                    loss.backward()
                    concept_optimizer.step()

                    train_loss += loss.detach().item()

            # Satisfaction of nf4 axioms:
            # C subClassOf R some D = \Forall x: C(x) \Implies R(x,f_R(x)) \sqcup D(f_R(x))
            # nf4_matrix_c.size() = n_concepts x n_constraints, Sparse matrix
            # nf4_matrix_R.size() = n_rels x n_constraints, Sparse matrix
            nf4_matrix_c, nf4_matrix_R, nf4_matrix_d = normal_form_matrix_dict['nf4']
            num_constraints = nf4_matrix_c.size(1)
            index_vec = th.arange(num_constraints)

            for batch_indices in index_data_loader:

                # batch_features = batch_features[0]
                batch_indices = batch_indices.to(device)

                batch_features = embedding_model(batch_indices)
                X_GO = concept_net(batch_features)

                f_R_mat = skolemizer(batch_features)  # |B| x n_rels x n_individuals
                i_max = f_R_mat.argmax(dim=2)  # get individuals with highest values i.e. argmax, |B| x n_rels
                sat_max = Exists(f_R_mat, dim=2,
                                 p=3)  # calculate Exists along individuals, see utils for more info, |B| x n_rels

                i_max_embedded = embedding_model(i_max)  # |B| x n_rels x latent_dim
                X_f_R = concept_net(i_max_embedded)  # |B| x n_rel x n_GO_classes

                X_R_x_f_R = relation_net(batch_features.unsqueeze(1).repeat(1, n_rels, 1),
                                         i_max_embedded).view(batch_size, -1)  # |B| x n_rels

                for i in range(0, num_constraints, axiom_batch_size):
                    batch_nf4_matrix_c = nf4_matrix_c.index_select(dim=1, index=index_vec[i:i + axiom_batch_size])
                    batch_nf4_matrix_R = nf4_matrix_R.index_select(dim=1, index=index_vec[i:i + axiom_batch_size])
                    batch_nf4_matrix_d = nf4_matrix_d.index_select(dim=1, index=index_vec[i:i + axiom_batch_size])

                    X_C = th.sparse.mm(X_GO, batch_nf4_matrix_c)  # |B| x axiom_batch_size
                    X_R =  th.sparse.mm(X_R_x_f_R, batch_nf4_matrix_R)  # |B| x axiom_batch_size
                    sat_max = th.sparse.mm(sat_max, batch_nf4_matrix_R)  # |B| x axiom_batch_size
                    X_R = X_R * sat_max  # include satisfaction from skolemizer

                    X_D = th.sparse.mm(X_f_R, batch_nf4_matrix_d)  # |B| x n_rel x axiom_batch_size
                    X_D = sparse_dense_mul(sparse_repeat(batch_nf4_matrix_R.unsqueeze(0), reps=batch_size, dim=0), X_D)
                    X_D = X_D.sum(dim=1).to_dense()  # |B| x axiom_batch_size

                    X_R_some_D = And(X_R, X_D)  # product t-norm through pointwise multiplication
                    out = Implies(X_C, X_R_some_D)  # Reichenbach Fuzzy Implication

                    out = ((out ** p).sum(dim=0) ** (1 / p)) / batch_size  # \Forall x
                    out = ((out ** p).sum() ** (1 / p)) / num_constraints  # \Forall constraints
                    loss = 1 - out
                    concept_optimizer.zero_grad()
                    loss.backward()
                    concept_optimizer.step()

                    train_loss += loss.detach().item()

            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for batch_features, batch_labels in train_loader:
                    bar.update(1)

                    batch_features = batch_features.to(device)
                    batch_features = encoder(batch_features).clone().detach()
                    batch_labels = batch_labels.to(device)

                    # Satisfaction of hasFunction relations:
                    # p subClassOf hasFunction some D


                    # relation_optimizer.step()
                    train_loss += 1-sat_agg
            
            train_loss /= train_steps
            
            print('Validation')
            # @TODO this is not done in the docs, why?
            concept_net.eval()
            relation_net.eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_labels) / batch_size))
                valid_loss = 0
                preds = []
                with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                    for batch_features, batch_labels in valid_loader:
                        bar.update(1)

                        batch_features = batch_features.to(device)
                        batch_features = encoder(batch_features).clone().detach()
                        batch_labels = batch_labels.to(device)

                        sat_agg = compute_batch_sat_level(batch_features=batch_features,
                                                          batch_labels=batch_labels,
                                                          terms_dict=terms_dict,
                                                          concept_constants=concept_constants,
                                                          relation_constants=rel_constants,
                                                          C_GO=C_go,
                                                          R_GO=R_go,
                                                          optimizers=[concept_optimizer, relation_optimizer],
                                                          normal_forms=normal_forms)

                        loss = 1 - sat_agg

                        valid_loss += loss
                        preds = np.append(preds, concept_net(batch_features).sigmoid().detach().cpu().numpy()[:, :n_terms])
                valid_loss /= valid_steps
                roc_auc = compute_roc(valid_labels, preds)
                print(f'Epoch {epoch}: Loss - {train_loss}, Valid loss - {valid_loss}, AUC - {roc_auc}')
                logger.writerow([epoch, train_loss, valid_loss, roc_auc])
            if valid_loss < best_loss:
                best_loss = valid_loss
                print('Saving model')
                th.save(C_go.model.state_dict(), model_file)

            concept_scheduler.step()
            # relation_scheduler.step()

        log_file.close()


    '''
    # Loading best model
    print('Loading the best model')
    net.load_state_dict(th.load(model_file))
    net.eval()
    with th.no_grad():
        test_steps = int(math.ceil(len(test_labels) / batch_size))
        test_loss = 0
        preds = []
        with ck.progressbar(length=test_steps, show_pos=True) as bar:
            for batch_features, batch_labels in test_loader:
                bar.update(1)
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                logits = net(batch_features)
                batch_loss = F.binary_cross_entropy(logits, batch_labels)
                test_loss += batch_loss.detach().cpu().item()
                preds = np.append(preds, logits.detach().cpu().numpy())
            test_loss /= test_steps
        preds = preds.reshape(-1, n_terms)
        roc_auc = compute_roc(test_labels, preds)
        print(f'Test Loss - {test_loss}, AUC - {roc_auc}')

    preds = list(preds)
    # Propagate scores using ontology structure
    for i in range(len(preds)):
        prop_annots = {}
        for go_id, j in terms_dict.items():
            score = preds[i][j]
            for sup_go in go.get_anchestors(go_id):
                if sup_go in prop_annots:
                    prop_annots[sup_go] = max(prop_annots[sup_go], score)
                else:
                    prop_annots[sup_go] = score
        for go_id, score in prop_annots.items():
            if go_id in terms_dict:
                preds[i][terms_dict[go_id]] = score

    test_df['preds'] = preds

    test_df.to_pickle(out_file)
    '''


def compute_batch_sat_level(batch_features,
                            batch_labels,
                            terms_dict,
                            concept_constants,
                            rel_constants,
                            C_go,
                            R_go,
                            optimizers,
                            normal_form_matrix_dict):
    batch_size = batch_features.size(0)
    return


    
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc


def build_normal_form_matrices(nb_GO_classes, n_rels, normal_forms):
    # pre-build matrices as masks for each constraint type for faster computation of constraint satisfaction
    # X_GO \in \mathbb{R}^{batch_size \times GO_class_individuals} is C_GO(x) for each row
    # Forall is defined w.r.t. LTNTorch (ltn.fuzzy_ops.AggregPMeanError)
    # *_M is matrix multiplication
    # .* is pointwise multiplication or other operator w.r.t. to considered t-norm
    nf1, nf2, nf3, nf4 = normal_forms

    return_matrices = {}

    # build nf1 with nf1_c/d \in \mathbb{R}^{GO_classes \times number_of_nf1_constraints}
    # For e.g. Product t-norm by Forall(X_GO *_M nf1_matrix_c .* X_GO *_M nf1_matrix_d)
    # Note that all matrices are sparse matrices
    n_constraints = len(nf1)
    column_indices = th.arange(n_constraints)
    nf1_matrix_c = th.sparse_coo_tensor(indices=th.cat((nf1[:, 0], column_indices)).reshape(2, -1),
                                        values=th.ones(n_constraints), size=(nb_GO_classes, n_constraints))
    nf1_matrix_d = th.sparse_coo_tensor(indices=th.cat((nf1[:, 1], column_indices)).reshape(2, -1),
                                        values=th.ones(n_constraints), size=(nb_GO_classes, n_constraints))
    return_matrices['nf1'] = [nf1_matrix_c, nf1_matrix_d]

    # nf2 similar to nf1 but with 3 masks for C,D and E
    n_constraints = len(nf2)
    column_indices = th.arange(n_constraints)
    nf2_matrix_c = th.sparse_coo_tensor(indices=th.cat((nf2[:, 0], column_indices)).reshape(2, -1),
                                        values=th.ones(n_constraints), size=(nb_GO_classes, n_constraints))
    nf2_matrix_d = th.sparse_coo_tensor(indices=th.cat((nf2[:, 1], column_indices)).reshape(2, -1),
                                        values=th.ones(n_constraints), size=(nb_GO_classes, n_constraints))
    nf2_matrix_e = th.sparse_coo_tensor(indices=th.cat((nf2[:, 2], column_indices)).reshape(2, -1),
                                        values=th.ones(n_constraints), size=(nb_GO_classes, n_constraints))
    return_matrices['nf2'] = [nf2_matrix_c, nf2_matrix_d, nf2_matrix_e]

    # nf3: R some C subClassOf D
    # for nf3 we only consider the matrix for the last concept D as R,C are existentially quantified
    n_constraints = len(nf3)
    column_indices = th.arange(n_constraints)
    nf3_matrix_R = th.sparse_coo_tensor(indices=th.cat((nf3[:, 0], column_indices)).reshape(2, -1),
                                        values=th.ones(n_constraints), size=(n_rels, n_constraints))
    nf3_matrix_c = th.sparse_coo_tensor(indices=th.cat((nf3[:, 1], column_indices)).reshape(2, -1),
                                        values=th.ones(n_constraints), size=(nb_GO_classes, n_constraints))
    nf3_matrix_d = th.sparse_coo_tensor(indices=th.cat((nf3[:, 2], column_indices)).reshape(2, -1),
                                        values=th.ones(n_constraints), size=(nb_GO_classes, n_constraints))
    return_matrices['nf3'] = [nf3_matrix_R, nf3_matrix_c, nf3_matrix_d]

    # nf4: C subClassOf R some D
    # for nf4 we only consider the matrix for the first concept C as R,D are existentially quantified
    n_constraints = len(nf4)
    column_indices = th.arange(n_constraints)
    nf4_matrix_c = th.sparse_coo_tensor(indices=th.cat((nf4[:, 0], column_indices)).reshape(2, -1),
                                        values=th.ones(n_constraints), size=(nb_GO_classes, n_constraints))
    nf4_matrix_R = th.sparse_coo_tensor(indices=th.cat((nf4[:, 1], column_indices)).reshape(2, -1),
                                        values=th.ones(n_constraints), size=(n_rels, n_constraints))
    nf4_matrix_d = th.sparse_coo_tensor(indices=th.cat((nf4[:, 2], column_indices)).reshape(2, -1),
                                        values=th.ones(n_constraints), size=(nb_GO_classes, n_constraints))

    return_matrices['nf4'] = [nf4_matrix_c, nf4_matrix_R, nf4_matrix_d]

    return return_matrices


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=True, layer_norm=True, dropout=0.3, activation=nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.activation(self.linear(x))
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class DGLTNConceptModel(nn.Module):
    def __init__(self, nb_iprs, nb_gos, device, nodes=[1024,]):
        super().__init__()
        self.nb_gos = nb_gos
        input_length = nb_iprs
        net = []
        for hidden_dim in nodes:
            net.append(MLPBlock(input_length, hidden_dim))
            net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
            input_length = hidden_dim
        net.append(nn.Linear(input_length, nb_gos))
        self.net = nn.Sequential(*net)

    def forward(self, features):
        features = self.net(features)
        return features.sigmoid()


class DGLTNRelationModel(nn.Module):
    def __init__(self, nb_iprs, nb_rels, device, nodes=[]):
        super().__init__()
        self.nb_rels = nb_rels
        input_length = nb_iprs
        net = []
        for hidden_dim in nodes:
            net.append(MLPBlock(input_length * 2, hidden_dim))
            net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
            input_length = hidden_dim
        net.append(nn.Linear(input_length, nb_rels))
        self.net = nn.Sequential(*net)

    def forward(self, x_features, y_features):
        return self.net(th.cat([x_features, y_features], dim=-1)).sigmoid()


class Skolemizer(nn.Module):
    def __init__(self, input_dim, nb_individuals, nb_rels, nodes=[512,]):
        super().__init__()
        self.nb_rels = nb_rels
        self.nb_individuals = nb_individuals

        input_length = input_dim
        net = []
        for hidden_dim in nodes:
            net.append(MLPBlock(input_length, hidden_dim))
            net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
            input_length = hidden_dim
        net.append(nn.Linear(input_length, nb_individuals * nb_rels))
        self.net = nn.Sequential(*net)

    def forward(self, x_features):
        return self.net(x_features).view(self.nb_rels, self.nb_individuals).sigmoid()


class ConceptLogitsToPredicate(nn.Module):
    def __init__(self, logits_model):
        super().__init__()
        self.logits_model = logits_model
        self.sigmoid = th.nn.Sigmoid()

    def forward(self, x, l):
        logits = self.logits_model(x)
        probs = self.sigmoid(logits)
        out = th.sum(probs * l, dim=1)
        return out


class RelationLogitsToPredicate(nn.Module):
    def __init__(self, logits_model):
        super().__init__()
        self.logits_model = logits_model
        self.sigmoid = th.nn.Sigmoid()

    def forward(self, x, y, l):
        logits = self.logits_model(x, y)
        probs = self.sigmoid(logits)
        out = th.sum(probs * l, dim=1)
        return out


class IndividualEmbedder(nn.Module):
    """
    A class to embed all individuals c_i associated with respective GO class C such that c_i\in C^\mathcal{I}_i.
    We hereby use latent_dim for both proteins and GO associated individuals to embed them into the same space.
    """
    def __init__(self, n_GO_classes, latent_dim):
        super().__init__()
        self.nb_gos = n_GO_classes
        self.embed = nn.Embedding(num_embeddings=n_GO_classes,
                                  embedding_dim=latent_dim)
        self.norm = nn.BatchNorm1d(latent_dim)
        k = math.sqrt(1/latent_dim)
        nn.init.uniform_(self.embed.weight, -k, k)
        self.all_gos = th.arange(n_GO_classes)

    def forward(self, indices):
        return self.norm(self.embed(indices))


class ConceptAutoencoder(nn.Module):
    def __init__(self, nb_iprs, latent_dim, nodes=[512,]):
        super().__init__()
        input_length = nb_iprs

        curr_dim = input_length
        encoder = []
        for hidden_dim in nodes:
            encoder.append(MLPBlock(input_length, hidden_dim))
            curr_dim = hidden_dim

        encoder.append(MLPBlock(curr_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder)

        curr_dim = latent_dim

        decoder = []
        for hidden_dim in nodes:
            decoder.append(MLPBlock(curr_dim, hidden_dim))
            curr_dim = hidden_dim

        decoder.append(nn.Linear(curr_dim, input_length))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, features):
        x = self.encoder(features)
        return self.decoder(x)


def load_normal_forms(go_file, terms_dict):
    nf1 = []
    nf2 = []
    nf3 = []
    nf4 = []
    relations = {}
    zclasses = {}

    def get_index(go_id):
        if go_id in terms_dict:
            index = terms_dict[go_id]
        elif go_id in zclasses:
            index = zclasses[go_id]
        else:
            zclasses[go_id] = len(terms_dict) + len(zclasses)
            index = zclasses[go_id]
        return index

    def get_rel_index(rel_id):
        if rel_id not in relations:
            relations[rel_id] = len(relations)
        return relations[rel_id]

    with open(go_file) as f:
        for line in f:
            line = line.strip().replace('_', ':')
            if line.find('SubClassOf') == -1:
                continue
            left, right = line.split(' SubClassOf ')
            # C SubClassOf D
            if len(left) == 10 and len(right) == 10:
                go1, go2 = left, right
                nf1.append((get_index(go1), get_index(go2)))
            elif left.find('and') != -1:  # C and D SubClassOf E
                go1, go2 = left.split(' and ')
                go3 = right
                nf2.append((get_index(go1), get_index(go2), get_index(go3)))
            elif left.find('some') != -1:  # R some C SubClassOf D
                rel, go1 = left.split(' some ')
                go2 = right
                nf3.append((get_rel_index(rel), get_index(go1), get_index(go2)))
            elif right.find('some') != -1:  # C SubClassOf R some D
                go1 = left
                rel, go2 = right.split(' some ')
                nf4.append((get_index(go1), get_rel_index(rel), get_index(go2)))
        return nf1, nf2, nf3, nf4, relations, zclasses


def load_data(data_root, ont, terms_file, load_encoded=True):
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    print('Terms', len(terms))

    ipr_df = pd.read_pickle(f'{data_root}/{ont}/interpros.pkl')
    iprs = ipr_df['interpros'].values
    iprs_dict = {v: k for k, v in enumerate(iprs)}

    if load_encoded:
        train_df = pd.read_pickle(f'{data_root}/{ont}/train_data_reduced.pkl')
        valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data_reduced.pkl')
        test_df = pd.read_pickle(f'{data_root}/{ont}/test_data_reduced.pkl')
    else:
        train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
        valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
        test_df = pd.read_pickle(f'{data_root}/{ont}/test_data.pkl')

    train_data = get_data(train_df, iprs_dict, terms_dict)
    valid_data = get_data(valid_df, iprs_dict, terms_dict)
    test_data = get_data(test_df, iprs_dict, terms_dict)

    return iprs_dict, terms_dict, train_data, valid_data, test_data, test_df


def get_data(df, iprs_dict, terms_dict):
    data = th.zeros((len(df), len(iprs_dict)), dtype=th.float32)
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    for i, row in enumerate(df.itertuples()):
        for ipr in row.interpros:
            if ipr in iprs_dict:
                data[i, iprs_dict[ipr]] = 1
        for go_id in row.prop_annotations: # prop_annotations for full model
            if go_id in terms_dict:
                g_id = terms_dict[go_id]
                labels[i, g_id] = 1
    return data, labels


if __name__ == '__main__':
    main()
