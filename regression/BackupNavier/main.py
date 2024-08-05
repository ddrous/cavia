import arguments
import cavia
import maml
import time

import os
import shutil

if __name__ == '__main__':

    args = arguments.parse_args()

    print("Using the following arguments:")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print()

    if args.task == 'sine':
        backup_folder = 'regression/BackupSine'
    elif args.task == 'celeba':
        backup_folder = 'regression/BackupCeleba'
    elif args.task == 'lotka':
        backup_folder = 'regression/BackupLotka'
    elif args.task == 'g_osci':
        backup_folder = 'regression/BackupGOsci'
    elif args.task == 'selkov':
        backup_folder = 'regression/BackupSelkov'
    elif args.task == 'brussel':
        backup_folder = 'regression/BackupBrussel'
    elif args.task == 'gray':
        backup_folder = 'regression/BackupGray'
    elif args.task == 'navier':
        backup_folder = 'regression/BackupNavier'

    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    shutil.copy('regression/main.py', backup_folder)
    shutil.copy('regression/arguments.py', backup_folder)
    shutil.copy('regression/cavia_model.py', backup_folder)
    shutil.copy('regression/cavia.py', backup_folder)
    shutil.copy('regression/logger.py', backup_folder)

    print(f"Copied files into to {backup_folder}")

    start = time.time()
    if args.maml:
        logger = maml.run(args, log_interval=10, rerun=True)
    else:
        logger = cavia.run(args, log_interval=1, rerun=True)
    end = time.time()

    print("\nTotal script time in hours minutes seconds: ", time.strftime("%H:%M:%S", time.gmtime(end - start)))