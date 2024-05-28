from invoke import task


@task()
def develop(c):
    c.run("python setup.py develop")
    c.run("pip install -e .[dev]")


@task()
def install(c):
    c.run("python setup.py install")


@task()
def test(c):
    c.run("pytest tests/")


@task
def testall(c):
    c.run("tox")


@task
def cover(c):
    c.run(
        "pytest tests -v --cov-report term --cov-report html:htmlcov --cov-report xml --cov=./src/"
    )
    c.run("coverage report")
    c.run("coverage xml")


@task
def quality(c):
    c.run("black --check src/ tests/")
    c.run("flake8 src/ tests/")


@task
def format(c):
    c.run("black src/ tests/")


@task
def clean(c):
    c.run("rm -rf target/* coverage* htmlcov/")


@task
def diagram(c):
    import plantuml
    from py2puml.py2puml import py2puml

    # writes the PlantUML content in a file
    with open("docs/class_diagram.plantuml", "w", encoding="utf8") as puml_file:
        puml_file.writelines(py2puml("src/LovelacePM", "LovelacePM"))

    p = plantuml.PlantUML("http://www.plantuml.com/plantuml/img/")
    p.processes_file("docs/class_diagram.plantuml", "docs/class_diagram.png")


@task
def doc(c, path=r"./docs"):
    import LovelacePM._version as version

    c.run("pip install pdoc")
    c.run(f'pdoc src/LovelacePM -o "{path}/latest"')
    c.run(f'pdoc src/LovelacePM -o "{path}/{version.version}"')
