"""
Flask application for the web interface.
"""
from flask import Flask
from app.web.routes import register_routes
from app.core.detector import PeopleDetector

def create_app(config=None):
    """
    Create and configure the Flask application
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Flask application instance
    """
    app = Flask(__name__, 
                template_folder='../../templates',
                static_folder='../../static')
    
    # Apply configuration
    if config:
        app.config.update(config)
    
    # Initialize detector
    detector = PeopleDetector(
        use_cpu=config.get('FORCE_CPU', False) if config else False,
        camera_url=config.get('CAMERA_URL') if config else None
    )
    
    # Start detection in background thread
    detector.start_detection()
    
    # Shutdown handler
    @app.teardown_appcontext
    def shutdown_detector(exception=None):
        detector.stop_detection()
    
    # Register routes
    register_routes(app, detector)
    
    return app
