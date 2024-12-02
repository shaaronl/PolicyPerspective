import casbin

def check_access(user, resource, action, enforcer):
    if enforcer.enforce(user, resource, action):
        print(f"user {user} is allowed to {action} on {resource}")
    else:
        print(f"user {user} is NOT allowed to {action} on {resource}")

def main():
    # example usage
    
    #initialize enforcer 
    enforcer = casbin.Enforcer("model.conf", "policy.csv")

        
    user = {"name" : "alice", "department": "finance"}
    resource = {"name": "report", "owner": "finance"}

    check_access(user, resource, "read", enforcer)
    check_access(user, resource, "write", enforcer)


if __name__ == "__main__":
    main()