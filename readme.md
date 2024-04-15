# fast_morse_smale_2d
WIP implementation for Morse-Smale complexes on regular 2D sampled data of Morse functions which utilizes the GPU. See the paper 'Parallel Computation of 2D Morse-Smale Complexes' by Shivashankar et. al.

Test data is obtained from [https://klacansky.com/open-scivis-datasets](https://klacansky.com/open-scivis-datasets).

Timing testing against [https://github.com/sci-visus/MSCEER/](https://github.com/sci-visus/MSCEER).

(TODO: proper citations.)

## Tests
If you wish to run the Python scripts which generate the testing data and visualize results, please install the requirements listed in `tests/requirements.txt`, as well as installing the `morse_smale` Python package from [https://github.com/uncommoncode/morse_smale](https://github.com/uncommoncode/morse_smale). You will need to install `wheel` using pip directly (not included in `requirements.txt`) and patch line 15 of `src/_cpy_morse_smale.pyx` to use `int` instead of `np.int`. (I have not had the time to create the patch myself, I simply edited the code manually.)
