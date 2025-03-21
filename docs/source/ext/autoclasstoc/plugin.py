# This contains code from package `autoclasstoc` (https://github.com/kalekundert/autoclasstoc)
# The original package follows the MIT license, see LICENSE.txt in the same folder
# It has been modified for the package `flory` by Yicheng Qiang.
# The `flory` package also follows the MIT license, see LICENSE in the root directory of the package.

from docutils.parsers.rst import directives
from sphinx.ext.autodoc import ClassDocumenter
from sphinx.util.docutils import SphinxDirective

from . import ConfigError, __version__, utils


class AutoClassToc(SphinxDirective):
    """
    Generate a succinct TOC for automatically documented classes.

    This class implements the :rst:dir:`autoclasstoc` directive.  More
    specifically, it implements the `run` function as expected by docutils.
    However, most of the actual logic is delegated to other classes and
    functions.
    """

    optional_arguments = 1
    option_spec = {
        "sections": utils.comma_separated_list,
        "exclude-sections": utils.comma_separated_list,
    }

    def run(self):
        """
        Create the nodes that will represent the class TOC.
        """
        try:
            qual_name = self.arguments[0] if self.arguments else None
            mod_name, cls_name = utils.pick_class(qual_name, self.env)
            mod, cls = utils.load_class(mod_name, cls_name)
            sections = utils.pick_sections(
                self.options.get("sections") or self.config.autoclasstoc_sections,
                exclude=self.options.get("exclude-sections"),
            )
            return utils.make_toc(self.state, cls, sections)

        except ConfigError as err:
            raise self.error(str(err))


def load_static_assets(app, config):
    """
    Add some rules for the spacing around <details> elements in class TOCs.
    """
    from pathlib import Path

    static_dir = Path(__file__).parent / "_static"
    static_dir = str(static_dir.resolve())

    if static_dir not in config.html_static_path:
        config.html_static_path.append(static_dir)

    app.add_css_file("autoclasstoc.css")


class AutoClassTocClassDocumenter(ClassDocumenter):
    option_spec = ClassDocumenter.option_spec.copy()
    option_spec["autoclasstoc"] = directives.flag

    def add_content(self, more_content):
        super().add_content(more_content)
        if "autoclasstoc" in self.options:
            self.add_line(".. autoclasstoc::", self.get_sourcename())


def setup(app):
    default_sections = [
        "public-attrs",
        "public-methods",
        "private-attrs",
        "private-methods",
    ]

    app.setup_extension("sphinx.ext.autosummary")
    app.add_config_value("autoclasstoc_sections", default_sections, "env")
    app.add_directive("autoclasstoc", AutoClassToc)
    app.add_autodocumenter(AutoClassTocClassDocumenter, override=True)
    app.connect("config-inited", load_static_assets)

    return {
        "version": __version__,
        "parallel_read_safe": True,
    }
