# coding=utf-8
import argparse
import logging
import os
import time

from nmmt import Suggestion


def translate_test(decoder):
#    suggestions = [
#        Suggestion('en', 'it', 'We offer a simple RESTful API', 'Offriamo una semplice API di tipo REST', 1.0),
#        Suggestion('en', 'it', 'The production system is running', 'Il sistema di produzione è in esecuzione', 1.0)]
    suggestions = []

    # Force full reset
    decoder._nn_needs_reset = True
    decoder._restorer._last_checkpoint = None

    result = decoder.translate('en', 'it',
                               suggestions=suggestions,
                               text='and')
#                               text='Companies and LSPs can translate their content with the ModernMT service')
#                               text='Companies and LSPs can translate their content with the ModernMT service '
#                                    'in many languages directly on their production environment '
#                                    'thanks to our simple RESTful API .')

    return result


def run_main():
    # Args parse
    # ------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Run a forever-loop testing translation speed')
    parser.add_argument('model', metavar='MODEL', help='the path to the decoder model')
    parser.add_argument('-g', '--gpu', dest='gpu', metavar='GPU', type=int, default=0,
                        help='the id of the GPU to use (default: 0)')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Setting up logging
    # ------------------------------------------------------------------------------------------------------------------
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Main loop
    # ------------------------------------------------------------------------------------------------------------------
    from nmmt.transformer import ModelConfig, TransformerDecoder
    from nmmt.checkpoint import CheckpointPool

    try:
        config = ModelConfig.load(args.model)

        # Init checkpoints
        begin_ts = time.time()
        builder = CheckpointPool.Builder()
        for name, checkpoint_path in config.checkpoints:
            builder.register(name, checkpoint_path)
        checkpoints = builder.build()
        logger.info('[1/2] Loaded %d checkpoints in %.1fs' % (len(checkpoints), time.time() - begin_ts))

        begin_ts = time.time()
        decoder = TransformerDecoder(args.gpu, checkpoints, config=config)
        logger.info('[2/2] Decoder created in %.1fs' % (time.time() - begin_ts))

        while True:
            result = translate_test(decoder)
            print result
            print result.text
            print result.alignment
#            break
    except KeyboardInterrupt:
        pass  # ignore and exit


if __name__ == '__main__':
    run_main()
