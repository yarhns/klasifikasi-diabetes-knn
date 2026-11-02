"""Application factory."""
from flask import Flask, render_template

from config import Config


def create_app(config_class=Config):
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )
    app.config.from_object(config_class)

    from app.routes import bp
    app.register_blueprint(bp)

    @app.errorhandler(404)
    def not_found(e):
        return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template("errors/500.html"), 500

    @app.errorhandler(413)
    def file_too_large(e):
        return "File terlalu besar (maks 5 MB).", 413

    return app