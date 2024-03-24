import arguments
import cavia
import maml
import time

if __name__ == '__main__':

    args = arguments.parse_args()

    start = time.time()
    if args.maml:
        logger = maml.run(args, log_interval=10, rerun=True)
    else:
        logger = cavia.run(args, log_interval=10, rerun=True)
    end = time.time()

    print("\nTotal script time in hours minutes seconds: ", time.strftime("%H:%M:%S", time.gmtime(end - start)))