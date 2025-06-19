from main import SessionLocal, User, get_password_hash

# Paramètres de l'utilisateur admin
username = "admin"
password = "admin123"  # À personnaliser
email = "admin@example.com"

# Création du hash du mot de passe
hashed_password = get_password_hash(password)

db = SessionLocal()
# Vérifier si l'utilisateur existe déjà
existing = db.query(User).filter(User.username == username).first()
if existing:
    print(f"L'utilisateur '{username}' existe déjà.")
else:
    user = User(username=username, password_hash=hashed_password, email=email, is_active=1)
    db.add(user)
    db.commit()
    print(f"Utilisateur admin '{username}' créé avec succès !")
db.close() 