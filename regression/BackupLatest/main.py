import arguments
import cavia
import maml
import time

if __name__ == '__main__':

    args = arguments.parse_args()



    print("Using the following arguments:")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    exit()




    start = time.time()
    if args.maml:
        logger = maml.run(args, log_interval=10, rerun=True)
    else:
        logger = cavia.run(args, log_interval=1, rerun=True)
    end = time.time()

    print("\nTotal script time in hours minutes seconds: ", time.strftime("%H:%M:%S", time.gmtime(end - start)))