
from api.database import engine, Base
from api.models import Review

try:
    Base.metadata.create_all(bind=engine)
    print("PostgreSQL connected successfully!")
    print("Reviews table created successfully!")

except Exception as e:
    print("Database setup failed:")
    print(e)
