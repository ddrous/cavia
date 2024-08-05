"""
Regression experiment using CAVIA
"""
import copy
import os
import time

import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F
import torch.optim as optim

import utils
import tasks_sine, tasks_celebA, tasks_selkov, tasks_lotka, tasks_g_osci, tasks_gray, tasks_brussel
from cavia_model import CaviaModel, CaviaModelOld, CaviaModelConv
from logger import Logger


## Print if CUDA is available
print("CUDA available: ", torch.cuda.is_available())


def run(args, log_interval=50, rerun=False):
    assert not args.maml

    # see if we already ran this experiment
    code_root = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isdir('{}/{}_result_files/'.format(code_root, args.task)):
        os.mkdir('{}/{}_result_files/'.format(code_root, args.task))
    path = '{}/{}_result_files/'.format(code_root, args.task) + utils.get_path_from_args(args)

    if os.path.exists(path + '.pkl') and not rerun:
        return utils.load_obj(path)

    start_time = time.time()
    utils.set_seed(args.seed)

    # --- initialise everything ---
    ode_tasks = ['selkov', 'lotka', 'g_osci', 'gray', 'brussel']

    # get the task family
    if args.task == 'sine':
        task_family_train = tasks_sine.RegressionTasksSinusoidal()
        task_family_valid = tasks_sine.RegressionTasksSinusoidal()
        task_family_test = tasks_sine.RegressionTasksSinusoidal()
    elif args.task == "selkov": # nohup python3 regression/main.py --task selkov --n_iter 100 --num_context_params 256 > nohup.log &
        task_family_train = tasks_selkov.RegressionTasksSelkov(mode='train')
        task_family_valid = tasks_selkov.RegressionTasksSelkov(mode='valid')
        task_family_adapt = tasks_selkov.RegressionTasksSelkov(mode='adapt')
        task_family_adapt_valid = tasks_selkov.RegressionTasksSelkov(mode='adapt_test')
    elif args.task == "lotka": # nohup python3 regression/main.py --task selkov --n_iter 100 --num_context_params 256 > nohup.log &
        task_family_train = tasks_lotka.RegressionTasksLotka(mode='train')
        task_family_valid = tasks_lotka.RegressionTasksLotka(mode='valid')
        task_family_test = tasks_lotka.RegressionTasksLotka(mode='adapt')
    elif args.task == 'g_osci':
        task_family_train = tasks_g_osci.RegressionTasksGOsci(mode='train')
        task_family_valid = tasks_g_osci.RegressionTasksGOsci(mode='valid')
        task_family_test = tasks_g_osci.RegressionTasksGOsci(mode='adapt')
    elif args.task == 'gray': # nohup python3 regression/main.py --task gray --n_iter 100 --num_context_params=1024 > nohup.log &
        task_family_train = tasks_gray.RegressionTasksGray(mode='train')
        task_family_valid = tasks_gray.RegressionTasksGray(mode='valid')
        task_family_adapt = tasks_gray.RegressionTasksGray(mode='adapt')
        task_family_adapt_valid = tasks_gray.RegressionTasksGray(mode='adapt_test')
    elif args.task == 'brussel': # nohup python3 regression/main.py --task brussel --n_iter 100 --num_context_params=256 > nohup.log &
        task_family_train = tasks_brussel.RegressionTasksBrussel(mode='train')
        task_family_valid = tasks_brussel.RegressionTasksBrussel(mode='valid')
        task_family_adapt = tasks_brussel.RegressionTasksBrussel(mode='adapt')
        task_family_adapt_valid = tasks_brussel.RegressionTasksBrussel(mode='adapt_test')
    elif args.task == 'celeba':
        task_family_train = tasks_celebA.CelebADataset('train', device=args.device)
        task_family_valid = tasks_celebA.CelebADataset('valid', device=args.device)
        task_family_test = tasks_celebA.CelebADataset('test', device=args.device)
    else:
        raise NotImplementedError

    # initialise network
    if args.task in ['selkov', 'lotka', 'g_osci']:
        model = CaviaModel(n_in=task_family_train.num_inputs,
                        n_out=task_family_train.num_outputs,
                        num_context_params=args.num_context_params,
                        n_hidden=args.num_hidden_layers,
                        device=args.device
                        ).to(args.device)
        n_training_tasks = len(task_family_train.environments)
    elif args.task in ['gray', 'brussel']:
        model = CaviaModelConv(n_in=task_family_train.num_inputs,
                        n_out=task_family_train.num_outputs,
                        num_context_params=args.num_context_params,
                        n_hidden=args.num_hidden_layers,
                        device=args.device
                        ).to(args.device)
        n_training_tasks = len(task_family_train.environments)
    else:
        model = CaviaModelOld(n_in=task_family_train.num_inputs,
                        n_out=task_family_train.num_outputs,
                        num_context_params=args.num_context_params,
                        n_hidden=args.num_hidden_layers,
                        device=args.device
                        ).to(args.device)
        n_training_tasks = args.tasks_per_metaupdate

    ## Count the number of parameters in the model
    print("Number of parameters in the model: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # We want 117502 total params for Brussel
    ## We want approximat
    # print("Number of parameters in the model: ", count_parameters(model, mode='ind'))
    # print("Number of environemtns: ", n_training_tasks, args.tasks_per_metaupdate)

    ## Print the model
    # print(model)

    # intitialise meta-optimiser
    # (only on shared params - context parameters are *not* registered parameters of the model)
    meta_optimiser = optim.Adam(model.parameters(), args.lr_meta)

    # initialise loggers
    logger = Logger()
    logger.best_valid_model = copy.deepcopy(model)

    # --- main training loop ---

    for i_iter in range(args.n_iter):

        # initialise meta-gradient
        meta_gradient = [0 for _ in range(len(model.state_dict()))]

        # sample tasks
        if args.task not in ode_tasks:
            target_functions = task_family_train.sample_tasks(n_training_tasks)

        # --- inner loop ---

        for t in range(n_training_tasks):

            torch.cuda.empty_cache()

            # reset private network weights
            model.reset_context_params()

            # get data for current task
            if args.task in ode_tasks:
                train_inputs, t_eval = task_family_train.sample_inputs(args.k_meta_train, t, args.use_ordered_pixels)
                train_inputs, t_eval = train_inputs.to(args.device), t_eval.to(args.device)
            else:
                train_inputs = task_family_train.sample_inputs(args.k_meta_train, args.use_ordered_pixels).to(args.device)


            # print("input:", train_inputs)
            # print("outputs:", train_targets)
            # import gc
            # gc.collect()
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))

            # train_inputs = train_inputs.to(args.device)
            # print(torch.cuda.memory_summary())

            for _ in range(args.num_inner_updates):
                # forward through model
                if args.task in ode_tasks:
                    train_outputs = model(train_inputs, t_eval)
                else:
                    train_outputs = model(train_inputs)

                # get targets
                if args.task in ode_tasks:
                    train_targets, t_eval = task_family_train.sample_targets(args.k_meta_train, t, args.use_ordered_pixels)
                    train_targets, t_eval = train_targets.to(args.device), t_eval.to(args.device)
                else:
                    train_targets = target_functions[t](train_inputs)
                    # train_targets = torch.Tensor(train_targets).to(args.device)


                # ------------ update on current task ------------

                # compute loss for current task
                task_loss = F.mse_loss(train_outputs, train_targets)

                # compute gradient wrt context params
                task_gradients = \
                    torch.autograd.grad(task_loss, model.context_params, create_graph=not args.first_order)[0]

                # update context params (this will set up the computation graph correctly)
                model.context_params = model.context_params - args.lr_inner * task_gradients

            # ------------ compute meta-gradient on test loss of current task ------------

            # get test data
            if args.task in ode_tasks:
                test_inputs, t_eval = task_family_train.sample_inputs(args.k_meta_test, t, args.use_ordered_pixels)
                test_inputs, t_eval = test_inputs.to(args.device), t_eval.to(args.device)
            else:
                test_inputs = task_family_train.sample_inputs(args.k_meta_test, args.use_ordered_pixels).to(args.device)

            # get outputs after update
            if args.task in ode_tasks:
                test_outputs = model(test_inputs, t_eval)
            else:
                test_outputs = model(test_inputs)

            # get the correct targets
            if args.task in ode_tasks:
                test_targets, t_eval = task_family_train.sample_targets(args.k_meta_test, t, args.use_ordered_pixels)
                test_targets, t_eval = test_targets.to(args.device), t_eval.to(args.device)
            else:
                test_targets = target_functions[t](test_inputs)
                # test_targets = torch.Tensor(test_targets).to(args.device)

            # compute loss after updating context (will backprop through inner loop)
            loss_meta = F.mse_loss(test_outputs, test_targets)

            # compute gradient + save for current task
            task_grad = torch.autograd.grad(loss_meta, model.parameters())

            for i in range(len(task_grad)):
                # clip the gradient
                meta_gradient[i] += task_grad[i].detach().clamp_(-10, 10)

        # ------------ meta update ------------

        # assign meta-gradient
        for i, param in enumerate(model.parameters()):
            param.grad = meta_gradient[i] / n_training_tasks

        # do update step on shared model
        meta_optimiser.step()

        # reset context params
        model.reset_context_params()


        # ------------ logging ------------

        if i_iter % log_interval == 0:

            # (re) adapt on the train set, then evaluate on the validation set
            loss_train, loss_valid, _ = eval_cavia(args, copy.deepcopy(model), task_family=task_family_train, task_family_test=task_family_valid,
                                              num_updates=args.num_inner_updates)
            logger.train_loss.append(loss_train)
            logger.test_loss.append(loss_valid)

            # adapt on adapatation train set, then evaluate on its validation set
            loss_adapt, loss_adapt_test, all_adapt_losses = eval_cavia(args, copy.deepcopy(model), task_family=task_family_adapt, task_family_test=task_family_adapt_valid,
                                              num_updates=args.num_inner_updates)
            logger.adapt_loss.append(loss_adapt)
            logger.adapt_loss_test.append(loss_adapt_test)
            logger.adapt_losses_test.append(all_adapt_losses)

            # # evaluate on adaptation set
            # loss_mean, loss_conf = eval_cavia(args, copy.deepcopy(model), task_family=task_family_test,
            #                                   num_updates=args.num_inner_updates)
            # logger.test_loss.append(loss_mean)
            # logger.test_conf.append(loss_conf)

            # save logging results
            utils.save_obj(logger, path)

            # save best model
            if logger.test_loss[-1] == np.min(logger.test_loss):
                print('saving best model at iter', i_iter)
                logger.best_valid_model = copy.deepcopy(model)

            # visualise results
            if args.task == 'celeba':
                task_family_train.visualise(task_family_train, task_family_test, copy.deepcopy(logger.best_valid_model),
                                            args, i_iter)

            # print current results
            logger.print_info(i_iter, start_time)
            start_time = time.time()

    return logger


def eval_cavia(args, model, task_family, task_family_test, num_updates, n_tasks=100, return_gradnorm=False):
    # get the task family
    ode_tasks = ['selkov', 'lotka', 'g_osci', 'gray', 'brussel']

    if args.task in ode_tasks:
        # all_inputs = task_family.data['X'].reshape((-1, task_family.num_inputs))
        pass
    else:
        # input_range = task_family.get_input_range().to(args.device)
        pass


    if args.task in ode_tasks:
        n_tasks = len(task_family.environments)

    # logging
    gradnorms = []

    # --- inner loop ---
    total_loss = 0

    losses_test = []
    total_loss_test = 0

    for t in range(n_tasks):

        # sample a task
        if args.task not in ode_tasks:
            target_function = task_family.sample_task()

        # reset context parameters
        model.reset_context_params()

        # get data for current task
        if args.task in ode_tasks:
            curr_inputs, t_eval = task_family.sample_inputs(args.k_shot_eval, t, args.use_ordered_pixels)
            curr_inputs, t_eval = curr_inputs.to(args.device), t_eval.to(args.device)

            curr_inputs_test, t_eval_test = task_family_test.sample_inputs(args.k_shot_eval, t, args.use_ordered_pixels)
            curr_inputs_test, t_eval_test = curr_inputs_test.to(args.device), t_eval_test.to(args.device)
        else:
            curr_inputs = task_family.sample_inputs(args.k_shot_eval, args.use_ordered_pixels).to(args.device)

            curr_inputs_test = task_family_test.sample_inputs(args.k_shot_eval, args.use_ordered_pixels).to(args.device)

        if args.task in ode_tasks:
            curr_targets, t_eval= task_family.sample_targets(args.k_shot_eval, t, args.use_ordered_pixels)
            curr_targets, t_eval = curr_targets.to(args.device), t_eval.to(args.device)

            curr_targets_test, t_eval_test = task_family_test.sample_targets(args.k_shot_eval, t, args.use_ordered_pixels)
            curr_targets_test, t_eval_test = curr_targets_test.to(args.device), t_eval_test.to(args.device)
        else:
            curr_targets = target_function(curr_inputs)

            curr_targets_test = target_function(curr_inputs_test)

        # ------------ update on current task ------------

        for _ in range(1, num_updates + 1):

            # forward pass
            # if args.task == "gray" and task_family.mode == "valid": ## Predict one after another, then concatenate
            #     curr_outputs = []
            #     for i in range(curr_inputs.shape[0]):
            #         curr_output = model(curr_inputs[i:i+1], t_eval)
            #         curr_outputs.append(curr_output)
            #     curr_outputs = torch.cat(curr_outputs, dim=1)
            if args.task in ['selkov', 'lotka', 'g_osci', 'gray', 'brussel']:
                curr_outputs = model(curr_inputs, t_eval)
            else:
                curr_outputs = model(curr_inputs)

            # compute loss for current task
            task_loss = F.mse_loss(curr_outputs, curr_targets)

            # compute gradient wrt context params
            task_gradients = \
                torch.autograd.grad(task_loss, model.context_params, create_graph=not args.first_order)[0]

            # update context params
            if args.first_order:
                model.context_params = model.context_params - args.lr_inner * task_gradients.detach()
            else:
                model.context_params = model.context_params - args.lr_inner * task_gradients

            # keep track of gradient norms
            gradnorms.append(task_gradients[0].norm().item())

        # ------------ logging ------------

        # compute true loss on entire input range
        model.eval()
        # losses.append(F.mse_loss(model(input_range), target_function(input_range)).detach().item())
        # losses.append(task_loss.detach().item())

        # ## Generate input and targets for all environments
        # if args.task in ode_tasks:
        #     all_inputs, t_eval = task_family.sample_inputs(args.k_shot_eval, t, args.use_ordered_pixels)
        #     all_inputs, t_eval = all_inputs.to(args.device), t_eval.to(args.device)
        #     all_targets, t_eval = task_family.sample_targets(args.k_shot_eval, t, args.use_ordered_pixels)
        #     all_targets, t_eval = all_targets.to(args.device), t_eval.to(args.device)
        # else:
        #     all_inputs = task_family.sample_inputs(args.k_shot_eval, args.use_ordered_pixels).to(args.device)
        #     all_targets = target_function(all_inputs)

        # ## Compute the loss for all environments
        # total_loss = 0
        # for i in range(len(all_inputs)):
        #     curr_inputs = all_inputs[i]
        #     curr_targets = all_targets[i]
        #     curr_outputs = model(curr_inputs, t_eval)

        #     curr_outputs = model(curr

        ## Eval on train set
        if args.task in ode_tasks:
            curr_outputs = model(curr_inputs, t_eval)
        else:
            curr_outputs = model(curr_inputs)

        ### Save the input and outputs if the task family is 'adapt' in npz
        if task_family.mode == 'adapt':
            if args.task in ode_tasks:
                np.savez("sample_predictions_cavia.npz", X_hat=curr_outputs.cpu().detach().numpy(), X=curr_targets.cpu().detach().numpy())

        task_loss = F.mse_loss(curr_outputs, curr_targets).detach().item()
        total_loss += task_loss

        ## Eval on test set
        if args.task in ode_tasks:
            curr_outputs_test = model(curr_inputs_test, t_eval_test)
        else:
            curr_outputs_test = model(curr_inputs_test)
        task_loss_test = F.mse_loss(curr_outputs_test, curr_targets_test).detach().item()
        losses_test.append(task_loss_test)
        total_loss_test += task_loss_test

        model.train()

    # losses = [total_loss / n_tasks]*5
    # losses_mean = np.mean(losses)
    # losses_conf = st.t.interval(0.95, len(losses) - 1, loc=losses_mean, scale=st.sem(losses))
    # if not return_gradnorm:
    #     return losses_mean, np.mean(np.abs(losses_conf - losses_mean))
    # else:
    #     return losses_mean, np.mean(np.abs(losses_conf - losses_mean)), np.mean(gradnorms)

    losses_mean = total_loss / n_tasks
    losses_mean_test = total_loss_test / n_tasks

    if not return_gradnorm:
        return losses_mean, losses_mean_test, losses_test
    else:
        return losses_mean, losses_mean_test, np.mean(gradnorms)

























# """
# Regression experiment using CAVIA
# """
# import copy
# import os
# import time

# import numpy as np
# import scipy.stats as st
# import torch
# import torch.nn.functional as F
# import torch.optim as optim

# import utils
# import tasks_sine, tasks_celebA
# from cavia_model import CaviaModel, CaviaModelOld
# from logger import Logger


# def run(args, log_interval=5000, rerun=False):
#     assert not args.maml

#     # see if we already ran this experiment
#     code_root = os.path.dirname(os.path.realpath(__file__))
#     if not os.path.isdir('{}/{}_result_files/'.format(code_root, args.task)):
#         os.mkdir('{}/{}_result_files/'.format(code_root, args.task))
#     path = '{}/{}_result_files/'.format(code_root, args.task) + utils.get_path_from_args(args)

#     if os.path.exists(path + '.pkl') and not rerun:
#         return utils.load_obj(path)

#     start_time = time.time()
#     utils.set_seed(args.seed)

#     # --- initialise everything ---

#     # get the task family
#     if args.task == 'sine':
#         task_family_train = tasks_sine.RegressionTasksSinusoidal()
#         task_family_valid = tasks_sine.RegressionTasksSinusoidal()
#         task_family_test = tasks_sine.RegressionTasksSinusoidal()
#     elif args.task == 'celeba':
#         task_family_train = tasks_celebA.CelebADataset('train', device=args.device)
#         task_family_valid = tasks_celebA.CelebADataset('valid', device=args.device)
#         task_family_test = tasks_celebA.CelebADataset('test', device=args.device)
#     else:
#         raise NotImplementedError

#     # initialise network
#     model = CaviaModelOld(n_in=task_family_train.num_inputs,
#                        n_out=task_family_train.num_outputs,
#                        num_context_params=args.num_context_params,
#                        n_hidden=args.num_hidden_layers,
#                        device=args.device
#                        ).to(args.device)

#     # intitialise meta-optimiser
#     # (only on shared params - context parameters are *not* registered parameters of the model)
#     meta_optimiser = optim.Adam(model.parameters(), args.lr_meta)

#     # initialise loggers
#     logger = Logger()
#     logger.best_valid_model = copy.deepcopy(model)

#     # --- main training loop ---

#     for i_iter in range(args.n_iter):

#         # initialise meta-gradient
#         meta_gradient = [0 for _ in range(len(model.state_dict()))]

#         # sample tasks
#         target_functions = task_family_train.sample_tasks(args.tasks_per_metaupdate)

#         # --- inner loop ---

#         for t in range(args.tasks_per_metaupdate):

#             # reset private network weights
#             model.reset_context_params()

#             # get data for current task
#             train_inputs = task_family_train.sample_inputs(args.k_meta_train, args.use_ordered_pixels).to(args.device)

#             for _ in range(args.num_inner_updates):
#                 # forward through model
#                 train_outputs = model(train_inputs)

#                 # get targets
#                 train_targets = target_functions[t](train_inputs)

#                 # ------------ update on current task ------------

#                 # compute loss for current task
#                 task_loss = F.mse_loss(train_outputs, train_targets)

#                 # compute gradient wrt context params
#                 task_gradients = \
#                     torch.autograd.grad(task_loss, model.context_params, create_graph=not args.first_order)[0]

#                 # update context params (this will set up the computation graph correctly)
#                 model.context_params = model.context_params - args.lr_inner * task_gradients

#             # ------------ compute meta-gradient on test loss of current task ------------

#             # get test data
#             test_inputs = task_family_train.sample_inputs(args.k_meta_test, args.use_ordered_pixels).to(args.device)

#             # get outputs after update
#             test_outputs = model(test_inputs)

#             # get the correct targets
#             test_targets = target_functions[t](test_inputs)

#             # compute loss after updating context (will backprop through inner loop)
#             loss_meta = F.mse_loss(test_outputs, test_targets)

#             # compute gradient + save for current task
#             task_grad = torch.autograd.grad(loss_meta, model.parameters())

#             for i in range(len(task_grad)):
#                 # clip the gradient
#                 meta_gradient[i] += task_grad[i].detach().clamp_(-10, 10)

#         # ------------ meta update ------------

#         # assign meta-gradient
#         for i, param in enumerate(model.parameters()):
#             param.grad = meta_gradient[i] / args.tasks_per_metaupdate

#         # do update step on shared model
#         meta_optimiser.step()

#         # reset context params
#         model.reset_context_params()

#         # ------------ logging ------------

#         if i_iter % log_interval == 0:

#             # evaluate on training set
#             loss_mean, loss_conf = eval_cavia(args, copy.deepcopy(model), task_family=task_family_train,
#                                               num_updates=args.num_inner_updates)
#             logger.train_loss.append(loss_mean)
#             logger.train_conf.append(loss_conf)

#             # evaluate on test set
#             loss_mean, loss_conf = eval_cavia(args, copy.deepcopy(model), task_family=task_family_valid,
#                                               num_updates=args.num_inner_updates)
#             logger.valid_loss.append(loss_mean)
#             logger.valid_conf.append(loss_conf)

#             # evaluate on validation set
#             loss_mean, loss_conf = eval_cavia(args, copy.deepcopy(model), task_family=task_family_test,
#                                               num_updates=args.num_inner_updates)
#             logger.test_loss.append(loss_mean)
#             logger.test_conf.append(loss_conf)

#             # save logging results
#             utils.save_obj(logger, path)

#             # save best model
#             if logger.valid_loss[-1] == np.min(logger.valid_loss):
#                 print('saving best model at iter', i_iter)
#                 logger.best_valid_model = copy.deepcopy(model)

#             # visualise results
#             if args.task == 'celeba':
#                 task_family_train.visualise(task_family_train, task_family_test, copy.deepcopy(logger.best_valid_model),
#                                             args, i_iter)

#             # print current results
#             logger.print_info(i_iter, start_time)
#             start_time = time.time()

#     return logger


# def eval_cavia(args, model, task_family, num_updates, n_tasks=100, return_gradnorm=False):
#     # get the task family
#     input_range = task_family.get_input_range().to(args.device)

#     # logging
#     losses = []
#     gradnorms = []

#     # --- inner loop ---

#     for t in range(n_tasks):

#         # sample a task
#         target_function = task_family.sample_task()

#         # reset context parameters
#         model.reset_context_params()

#         # get data for current task
#         curr_inputs = task_family.sample_inputs(args.k_shot_eval, args.use_ordered_pixels).to(args.device)
#         curr_targets = target_function(curr_inputs)

#         # ------------ update on current task ------------

#         for _ in range(1, num_updates + 1):

#             # forward pass
#             curr_outputs = model(curr_inputs)

#             # compute loss for current task
#             task_loss = F.mse_loss(curr_outputs, curr_targets)

#             # compute gradient wrt context params
#             task_gradients = \
#                 torch.autograd.grad(task_loss, model.context_params, create_graph=not args.first_order)[0]

#             # update context params
#             if args.first_order:
#                 model.context_params = model.context_params - args.lr_inner * task_gradients.detach()
#             else:
#                 model.context_params = model.context_params - args.lr_inner * task_gradients

#             # keep track of gradient norms
#             gradnorms.append(task_gradients[0].norm().item())

#         # ------------ logging ------------

#         # compute true loss on entire input range
#         model.eval()
#         losses.append(F.mse_loss(model(input_range), target_function(input_range)).detach().item())
#         model.train()

#     losses_mean = np.mean(losses)
#     losses_conf = st.t.interval(0.95, len(losses) - 1, loc=losses_mean, scale=st.sem(losses))
#     if not return_gradnorm:
#         return losses_mean, np.mean(np.abs(losses_conf - losses_mean))
#     else:
#         return losses_mean, np.mean(np.abs(losses_conf - losses_mean)), np.mean(gradnorms)