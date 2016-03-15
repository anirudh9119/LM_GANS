- Install platoon
- Install mimir (https://github.com/bartvm/mimir)

Discriminator code needs to be refactored. 

To launch the code

- python lm_contoller.py config.json in one terminal 

and THEANO_FLAGS=device=gpu0, floatX=float32 python lm_worker.py
