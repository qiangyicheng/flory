# OUTPUT_NAME must always be paper
INPUT_NAME = paper
OUTPUT_NAME = paper

ALL_OUTPUTS = $(OUTPUT_NAME).pdf $(OUTPUT_NAME).crossref

all: $(ALL_OUTPUTS)

$(ALL_OUTPUTS): $(INPUT_NAME).md $(INPUT_NAME).bib
	@docker run --rm -it -v $(PWD):/data -u $(id -u):$(id -g) openjournals/inara -o pdf,crossref $(INPUT_NAME).md

.PHONY: clean

clean:
	@rm -f ${ALL_OUTPUTS} 