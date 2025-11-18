# build single *.cu program or run bin/${NAME} on GPGPU-Sim at specified config under spack env
# GPGPUSIM : name of GPGPU-Sim on spack
# CONFIG : which config on GPGPU-Sim eg. RTX2060 GTX480 TITANV
# NAME : ${SRC}/${NAME}.cu or ${BIN}/${NAME}
# OUTNAME : part name of GPGPU-Sim generated file 
# BIN : dir where binary file lie
# OUT : dir where generated file lie
# SRC : dir where source file lie
# SIM : dir where execute simulating
# IFBUILD : 1 not skip build ; 0 skip build and just use bin/${NAME}
# IFBACKGROUND : 1 background ; 0 foreground
# ARG : arg pass to the program
# ARCH : nvcc -arch=${ARCH} use for build 
# IFDEBUG : 1 use gdb and source debug in GPGPUSIM; 0 not use gdb
# CONFIG_SELECT : 0 only build sim env 1 run 2 run rebuild env
# IFLINKDATA : 1 link DATADIR ; 0 not link DATADIR
# DATADIR : the data directory when the program runs

NAME=test
CONFIG=RTX3070 # Using RTX3070 config for RTX 3060Ti (both SM86)
ARCH=sm_86
IFBUILD=1
IFBACKGROUND=0
IFDEBUG=0
CONFIG_SELECT=2 # 0 only build sim env ;1 run ;2 run rebuild env
ARG=
IFLINKDATA=0
DATADIR=

OUTNAME=${NAME}

BIN=bin
OUT=out
SRC=src
SIM=sim

CURDIR=$(pwd)

# GPGPU-Sim distribution path (assuming it's two directories up from deAttn)
GPGPUSIM_ROOT=$(cd "${CURDIR}/.." && pwd)

OUTPATH=${CURDIR}/${OUT}/${OUTNAME}_${CONFIG}.txt
SIMPATH=${CURDIR}/${SIM}/${OUTNAME}_${CONFIG}
EXEPATH=${CURDIR}/${BIN}/${NAME}

[ ! -d ${SRC} ] && mkdir ${SRC}

if [ ! -e ${SRC}/${NAME}.cu ] && [ ${IFBUILD} -eq 1 ]  # test *.cu exist
then
	echo ${SRC}/${NAME}.cu "not exists"
elif [ ! -e ${BIN}/${NAME} ] && [ ${IFBUILD} -eq 0 ] # test binary exist
then
	echo ${BIN}/${NAME} "not exists"
elif [ ! -d ${GPGPUSIM_ROOT}/configs/tested-cfgs/SM*_${CONFIG} ]
then # test config exists
	echo "config" ${CONFIG} "not exists"
	ls ${GPGPUSIM_ROOT}/configs/tested-cfgs
elif [ ! -d ${SIMPATH} ] && [ ${CONFIG_SELECT} -eq 1 ]
then # test SIMPATH exists
	echo "build sim env first"
else
	#init dir
	[ ! -d ${SIM} ] && mkdir ${SIM}
	[ ! -d ${BIN} ] && mkdir ${BIN}
	[ ! -d ${OUT} ] && mkdir ${OUT}

	if [ ${CONFIG_SELECT} -ne 1 ]
	then
		echo "build sim env"
		rm -rf ${SIMPATH}
		cp -r ${GPGPUSIM_ROOT}/configs/tested-cfgs/SM*_${CONFIG} ${SIMPATH}
	else
		echo "use existed sim env"
	fi

	if [ ${IFBUILD} -eq 1 ] 
	then
		echo "build" ${NAME}.cu
		# 只编译测试程序
		nvcc -arch=${ARCH} --cudart shared -I${SRC} \
		     ${SRC}/${NAME}.cu \
		     -o ${BIN}/${NAME}
	else 
		echo "skip build from source"
	fi

	# Setup GPGPU-Sim environment
	export CUDA_INSTALL_PATH=$(dirname $(dirname $(which nvcc)))
	if [ ${IFDEBUG} -eq 0 ]; then
		. ${GPGPUSIM_ROOT}/setup_environment 2>&1 1>&/dev/null
	else
		. ${GPGPUSIM_ROOT}/setup_environment debug 2>&1 1>&/dev/null
	fi 

	if [ ${IFLINKDATA} -eq 1 ]; then
		ln -s ${DATADIR} ${SIMPATH}
	fi

	if [ ${CONFIG_SELECT} -ne 0 ]
	then
		echo "execute" ${NAME} ${ARG} "on GPGPU-Sim with" ${CONFIG} "config"
	fi

	if [ ${IFDEBUG} -eq 1 ] && [ ${CONFIG_SELECT} -ne 0 ]
	then
		cd ${SIMPATH} && gdb ${EXEPATH} 
	elif [ ${IFBACKGROUND} -eq 1 ] && [ ${CONFIG_SELECT} -ne 0 ]
	then
		cd ${SIMPATH} && \
		nohup ${EXEPATH} ${ARG} > ${OUTPATH} 2>&1 & # run at background
	elif [ ${CONFIG_SELECT} -ne 0 ]
	then
		cd ${SIMPATH} && \
		${EXEPATH} ${ARG} > ${OUTPATH} # run at foreground
	fi
	cd ${CURDIR}
fi

