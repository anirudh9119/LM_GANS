THEANO_FLAGS="device=gpu0" python lm_controller.py config_forecasting.json &
THEANO_FLAGS="device=gpu0" python lm_char_worker.py &

