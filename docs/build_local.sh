#!/bin/bash


DIR=$(dirname ${BASH_SOURCE})
if [[ -z "${DIR}" ]]; then
	echo "Directory not found correctly"
fi

rm -r "${DIR}/_build/html"
rm -r "${DIR}/autoapi"


sphinx-build "${DIR}" "${DIR}/_build/html"